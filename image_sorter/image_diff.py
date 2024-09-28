#!/opt/homebrew/bin/python3

# import the necessary packages
import argparse
from typing import Callable, Optional, TypedDict

import cv2
import imutils
import skimage
from cv2.typing import MatLike
from PIL import Image
from skimage.metrics import structural_similarity

np = cv2.numpy


def get_image(f: str) -> MatLike:
    return Image.fromarray(skimage.io.imread(f))


class DiffResult(TypedDict):
    diff: Optional[MatLike]
    score: Optional[float]
    pctArea: Optional[float]


def _SSIM(imageA: MatLike, imageB: MatLike, visualize=False) -> DiffResult:
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    areaA = grayA.shape[0] * grayA.shape[1]
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    if grayA.shape != grayB.shape:
        (score, diff, pctArea) = 0, None, 0
    else:
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
    # print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        pctArea = 1 - sum(cv2.contourArea(cnt) for cnt in cnts) / areaA
    if visualize:
        # loop over the contours
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
    return _SSIM(cv2.imread(imageA), cv2.imread(imageB), visualize=visualize)


def _pixelwise(imageA: MatLike, imageB: MatLike, visualize=False, cutoff=None) -> DiffResult:
    if imageA.shape != imageB.shape:
        return {'diff': None, 'score': 0, 'pctArea': 0}
    if cutoff is None:
        mask_diff = np.tensordot(cv2.absdiff(imageA, imageB), [1, 1, 1], axes=1)
        score = pctArea = 1 - np.count_nonzero(mask_diff) / mask_diff.size
    else:
        size = imageA.size
        length = imageA.shape[0]
        stepsize = max(50, int(length * cutoff))
        Area = 0
        threshold = size * cutoff
        score = pctArea = 0.
        for r in (slice(i, i + stepsize) for i in range(0, length, stepsize)):
            subimageA, subimageB = imageA[r], imageB[r]
            mask_diff = np.tensordot(cv2.absdiff(subimageA, subimageB), [1, 1, 1], axes=1)
            Area += mask_diff.size - np.count_nonzero(mask_diff)
            if Area > threshold:
                score = pctArea = 1.
                break
    return {'diff': None, 'score': score, 'pctArea': pctArea}


def pixelwise(imageA: str, imageB: str, visualize=False, **kwds) -> DiffResult:
    return _pixelwise(cv2.imread(imageA), cv2.imread(imageB), visualize=visualize, **kwds)


class _CCIP:

    _metric: Callable[[MatLike, MatLike], float] = lambda x, y: 0.

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

    _metric: Callable[[MatLike, MatLike], float] = lambda x, y: 0.

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
