#!/opt/homebrew/bin/python3

# import the necessary packages
import argparse
from typing import Callable, Optional, TypeAlias, TypedDict

import numpy as np
from PIL import Image, ImageOps
import imageio


MatLike: TypeAlias = np.ndarray[tuple[int, int, int, int], np.dtype[np.float64 | np.int_]]


def get_image(f: str) -> Image.Image:
    return Image.fromarray(imageio.imread(f))


class DiffResult(TypedDict):
    diff: Optional[MatLike]
    score: Optional[float]
    pctArea: Optional[float]


def _SSIM(imageA: Image.Image, imageB: Image.Image, visualize=False) -> DiffResult:
    from skimage.metrics import structural_similarity

    # convert the images to grayscale
    grayA = ImageOps.grayscale(imageA)
    grayB = ImageOps.grayscale(imageB)

    areaA = grayA.width * grayA.height
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    if grayA.size != grayB.size:
        (score, diff, pctArea) = 0, None, 0
    else:
        score, diff = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        # threshold the difference image
        thresh = diff > np.percentile(diff, 99)
        pctArea = 1. - np.sum(thresh) / areaA
    if visualize:
        import cv2
        import imutils
        # loop over the contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # show the output images
        cv2.imshow("Original", imageA)
        cv2.imshow("Modified", imageB)
        cv2.imshow("Diff", diff)
        cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)
    return {'diff': diff, 'score': score, 'pctArea': pctArea}


def SSIM(imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
    return _SSIM(get_image(imageA), get_image(imageB), visualize=visualize)


def _pixelwise(imageA: Image.Image, imageB: Image.Image, visualize=False, cutoff=None) -> DiffResult:
    if imageA.size != imageB.size:
        return {'diff': None, 'score': 0, 'pctArea': 0}
    arr_imageA = np.asarray(imageA)
    arr_imageB = np.asarray(imageB)
    if cutoff is None:
        mask_diff = np.tensordot(abs(arr_imageA - arr_imageB), [1, 1, 1], axes=1)
        score = pctArea = 1 - np.count_nonzero(mask_diff) / mask_diff.size
    else:
        size = imageA.size
        length = imageA.size[0]
        stepsize = max(50, int(length * cutoff))
        Area = 0
        threshold = size * cutoff
        score = pctArea = 0.
        for r in (slice(i, i + stepsize) for i in range(0, length, stepsize)):
            subimageA, subimageB = arr_imageA[r], arr_imageB[r]
            mask_diff = np.tensordot(abs(subimageA - subimageB), [1, 1, 1], axes=1)
            Area += mask_diff.size - np.count_nonzero(mask_diff)
            if Area > threshold:
                score = pctArea = 1.
                break
    return {'diff': None, 'score': score, 'pctArea': pctArea}


def pixelwise(imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
    return _pixelwise(get_image(imageA), get_image(imageB), visualize=visualize, **kwds)


class _CCIP:

    _metric: Callable[[Image.Image, Image.Image], float] = lambda x, y: 0.

    def __call__(self, imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
        return self._firt_call(imageA, imageB, visualize, **kwds)

    @classmethod
    def _firt_call(cls, imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
        import imgutils.metrics as metrics
        cls._metric = metrics.ccip_same
        cls.__call__ = cls._call
        return cls._call(imageA, imageB, visualize, **kwds)

    @classmethod
    def _call(cls, imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
        score = float(cls._metric(get_image(imageA), get_image(imageB)))
        return {'diff': None, 'score': score, 'pctArea': None}


ccip = _CCIP()


class _LPIPS:

    _metric: Callable[[Image.Image, Image.Image], float] = lambda x, y: 0.

    def __call__(self, imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
        return self._firt_call(imageA, imageB, visualize, **kwds)

    @classmethod
    def _firt_call(cls, imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
        import imgutils.metrics as metrics
        cls._metric = metrics.lpips_difference
        cls.__call__ = cls._call
        return cls._call(imageA, imageB, visualize, **kwds)

    @classmethod
    def _call(cls, imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
        score = 1. - float(cls._metric(get_image(imageA), get_image(imageB)))
        return {'diff': None, 'score': score, 'pctArea': None}


lpips = _LPIPS()


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs=2, help="input images")
    ap.add_argument("--visualize", '-V', action='store_true', help="visualize")
    args = ap.parse_args()

    # load the two input images
    d = SSIM(*args.images, visualize=args.visualize)


if __name__ == '__main__':
    main()
