#!/opt/homebrew/bin/python3

# import the necessary packages
import argparse
import re
from typing import Callable, Optional, TypedDict


import numpy as np
from PIL import Image, ImageOps
from . import element


class DiffResult(TypedDict):
    score: Optional[float]


def rescaled(image: Image.Image, image_ref: Image.Image) -> Image.Image:
    return image.resize(image_ref.size)


class Differ[F: tuple[element.Features, ...]]:
    def __init__(
        self,
        context: element.ClassificationContext,
        _compute_features: Optional[Callable[[Image.Image], F]] = None,
        _compute_score: Optional[Callable[[F, F], float]] = None,
        cached_features: bool = True
    ):
        self.context: element.ClassificationContext = context
        self._compute_features: Optional[Callable[[Image.Image], F]] = _compute_features
        self._compute_score: Optional[Callable[[F, F], float]] = _compute_score
        self.cached_features: bool = cached_features

    @classmethod
    def get_model(cls):
        ...

    @classmethod
    def get_feature_shapes(cls) -> tuple[tuple[int, ...], ...] | None:
        m = cls.get_model()
        if m is None:
            return None
        return tuple(
            tuple(
                d if isinstance(d, int) else -1 for d in out.shape
            ) for out in m.outputs()
        )

    @classmethod
    def get_feature_dtype(cls) -> np.dtype:
        m = cls.get_model()
        if m is None:
            return np.dtype(np.float32)
        outputs = m.outputs()
        dtype_str = outputs[0].dtype
        assert all(dtype_str == out.dtype for out in outputs[1:]), "Inconsistent output dtypes"
        # Mapping dictionary for common types
        type_map = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
            'tensor(double)': np.float64,
            'tensor(int64)': np.int64,
            'tensor(int32)': np.int32,
            'tensor(bool)': np.bool_
        }
        return np.dtype(type_map.get(dtype_str, np.float32))

    @classmethod
    def create_empty_features(cls) -> element.SharedNDArray | None:
        dtype = cls.get_feature_dtype()
        shapes = cls.get_feature_shapes()
        if shapes is None:
            return None
        return element.SharedNDArray(shapes=shapes, dtype=dtype)

    def prepare_features(self, e: element.Element) -> element.SharedNDArray | None:
        shared_features: element.SharedNDArray = e.get(self.context, 'features')
        if shared_features is None:
            shared_features = self.create_empty_features()
            if shared_features is not None and self.cached_features:
                e.set(self.context, features=shared_features)
        return shared_features

    def compute_features(self, e: element.Element) -> F:
        shared_features = self.prepare_features(e)

        if shared_features is not None:
            shared_features.update_from_worker(
                lambda: self._compute_features(e.get_image())  # pyright: ignore[reportOptionalCall]
            )
            features = tuple(shared_features.as_ndarrays())
        else:
            features = self._compute_features(e.get_image()) # pyright: ignore[reportOptionalCall]

        return features

    def compute_score(self, e1: element.Element, e2: element.Element) -> float:
        f1 = self.compute_features(e1)
        f2 = self.compute_features(e2)
        return self._compute_score(f1, f2)  # pyright: ignore[reportOptionalCall]

    def init(self) -> None:
        ...

    def __call__(self, imageA: element.InputLike, imageB: element.InputLike, **kwds) -> DiffResult:
        return self._first_call(imageA, imageB, **kwds)

    def _first_call(
        self, imageA: element.InputLike, imageB: element.InputLike, **kwds
    ) -> DiffResult:
        self.init()
        self.__call__ = self._call
        return self._call(imageA, imageB, **kwds)

    def _call(self, imageA: element.InputLike, imageB: element.InputLike, **kwds) -> DiffResult:
        imageA = element.Element.as_element(imageA)
        imageB = element.Element.as_element(imageB)
        score = self.compute_score(imageA, imageB)
        return DiffResult(score=score)


class Pixelwise(Differ):

    def __init__(
        self,
        context: element.ClassificationContext = element.ClassificationContext(alg='pixelwise'),
        cached_features: bool = False,
        cutoff: Optional[float] = None,
        **kwds
    ):
        assert context.alg == 'pixelwise'
        super().__init__(context, None, None, cached_features=cached_features)
        self.cutoff = cutoff
        self.kwds = kwds

    def init(self):
        def _compute_features(img: Image.Image):
            return (np.asarray(img),)

        def _compute_score(f1: tuple[element.Features, ...], f2: tuple[element.Features, ...]) -> float:
            cutoff = self.cutoff
            imageA = f1[0]
            imageB = f2[0]
            if (imageA.shape != imageB.shape):
                return 0.
            b = np.ones(imageA.shape[-1])
            if cutoff is None:
                mask_diff = np.tensordot(abs(imageA - imageB), b, axes=1)
                pctArea = 1 - np.count_nonzero(mask_diff) / mask_diff.size
            else:
                size = np.prod(imageA.shape[:2])
                length = imageA.shape[1]
                stepsize = max(1, int(length * cutoff))
                Area = 0
                threshold = size * cutoff
                pctArea = 0.
                for r in (slice(i, i + stepsize) for i in range(0, length, stepsize)):
                    subimageA, subimageB = imageA[r], imageB[r]
                    mask_diff = np.tensordot(abs(subimageA - subimageB), [1, 1, 1], axes=1)
                    Area += mask_diff.size - np.count_nonzero(mask_diff)
                    if Area > threshold:
                        pctArea = 1.
                        break
            return pctArea
        self._compute_features = _compute_features
        self._compute_score = _compute_score


pixelwise = Pixelwise()


class _SSIM(Differ):

    def __init__(
        self,
        context: element.ClassificationContext = element.ClassificationContext(alg='ssim'),
        cached_features: bool = False,
        autoscale: bool = True,
        **kwds
    ):
        assert context.alg == 'ssim'
        super().__init__(context, None, None, cached_features=cached_features)
        self.autoscale = autoscale
        self.kwds = kwds

    def init(self):
        from skimage.metrics import structural_similarity
        def _compute_features(img: Image.Image):
            gray = ImageOps.grayscale(img)
            return (np.asarray(gray),)

        def _compute_score(f1: tuple[element.Features, ...], f2: tuple[element.Features, ...]) -> float:
            score = 0.
            grayA, grayB = f1[0], f2[0]
            if self.autoscale:
                if grayA.shape != grayB.shape:
                    grayB = np.asarray(rescaled(Image.fromarray(grayB), Image.fromarray(grayA)))
            else:
                return score
            score, _ = structural_similarity(grayA, grayB, full=True)
            return score

        self._compute_features = _compute_features
        self._compute_score = _compute_score


SSIM = _SSIM()


class _CCIP(Differ[tuple[element.Features, ...]]):

    def __init__(self, context: element.ClassificationContext = element.ClassificationContext(alg='ccip')):
        assert context.alg == 'ccip'
        super().__init__(context, None, None)

    def init(self):
        from imgutils.metrics import ccip_extract_feature, ccip_same

        def _compute_features(img: Image.Image) -> tuple[element.Features, ...]:
            return (ccip_extract_feature(img),)

        def _compute_score(f1: tuple[element.Features, ...], f2: tuple[element.Features, ...]) -> float:
            return ccip_same(f1[0], f2[0])

        self._compute_features = _compute_features
        self._compute_score = _compute_score

    @classmethod
    def get_model(cls):
        import imgutils.metrics.ccip
        m = imgutils.metrics.ccip._open_feat_model(
        imgutils.metrics.ccip._DEFAULT_MODEL_NAMES
        )
        return m


ccip = _CCIP()


class _LPIPS(Differ[tuple[element.Features, ...]]):

    def __init__(self, context: element.ClassificationContext = element.ClassificationContext(alg='lpips')):
        assert context.alg == 'lpips'
        super().__init__(context, None, None)

    def init(self):
        from imgutils.metrics import lpips_extract_feature, lpips_difference

        def _compute_features(img: Image.Image) -> tuple[element.Features, ...]:
            return lpips_extract_feature(img)

        def _compute_score(f1: tuple[element.Features, ...], f2: tuple[element.Features, ...]) -> float:
            return 1. - float(lpips_difference(f1, f2))

        self._compute_features = _compute_features
        self._compute_score = _compute_score

    @classmethod
    def get_model(cls):
        import imgutils.metrics.lpips
        m = imgutils.metrics.lpips._lpips_feature_model()
        return m


lpips = _LPIPS()


alg_map: dict[str, Differ] = {
    'pixelwise': pixelwise,
    'ssim': SSIM,
    'ccip': ccip,
    'lpips': lpips
}


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(
        description="Compute the Similarity between two images using various algorithms."
    )
    ap.add_argument("images", nargs=2, help="input images")
    ap.add_argument(
        "--alg",
        type=str,
        default="pixelwise",
        choices=["pixelwise", "ssim", "ccip", "lpips"],
        help="difference algorithm to use"
    )
    args = ap.parse_args()
    differ = alg_map[args.alg]
    score = differ(*args.images)['score']
    print(f"🔍 Algorithm: {args.alg}")
    print(f"📸 Images: `{args.images[0]}` ↔ `{args.images[1]}`")
    print(f"📊 Similarity Score: {score:.2f}")


if __name__ == '__main__':
    main()
