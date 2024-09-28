#!/opt/homebrew/bin/python3

import bisect
import concurrent.futures
from pathlib import Path
import mimetypes
from itertools import groupby
from functools import cache, partial
from typing import Callable, Iterable, Iterator, Literal, Optional, Sequence, TypeVar, TypedDict
import easy_logging
from .image_diff import SSIM, DiffResult, pixelwise, ccip, lpips


logger = easy_logging.get_logger('ImageSorter')


def get_all_images(path: Sequence[Path]) -> Iterator[Path]:
    if len(path) == 1 and path[0].is_dir():
        flist = path[0].iterdir()
    else:
        flist = path
    for f in flist:
        mime = mimetypes.guess_type(f)[0]
        if mime is not None:
            if mime.split('/')[0] == 'image':
                yield f


T = TypeVar('T')


def compare(
    original: str,
    compared: str,
    alg: Callable[[str, str], DiffResult],
    target: str,
    threshold: float
) -> bool:
    return (alg(original, compared)[target] > threshold)


def get_grouped(
    original: T,
    *compared: T,
    alg: Callable[[str, str], DiffResult] = pixelwise,
    threshold: float = 0.1,
    key: Optional[Callable[[T], str]] = None
) -> list[T]:

    if len(compared) == 0:
        return [original]

    if key is not None:
        _original = key(original)
        _compared = map(key, compared)
    else:
        _original = original
        _compared = compared

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(
            partial(compare, compared=str(_original), alg=alg, target='pctArea', threshold=threshold),
            map(str, _compared)
        )
        return [original, *(compared[i] for i, r in enumerate(results) if r)]


def _compare_groups_bruteforce(
    group_a: Sequence[T],
    *groups_b: Sequence[T],
    key: Optional[Callable[[T], str]] = None,
    alg: Optional[str] = None,
    target: str = 'pctArea',
    threshold: Optional[float] = None
) -> list[bool]:
    _group_a = group_a
    _groups_b = groups_b
    if key is not None:
        _group_a = map(key, _group_a)
        _groups_b = (map(key, _group_b) for _group_b in _groups_b)

    if alg is None:
        alg = 'pixelwise'
    fn = ImageSorter.algs[alg]
    if threshold is None:
        threshold = ImageSorter.thresholds[alg]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results: list[list[Iterator[bool]]] = []
        for _group_b in _groups_b:
            result_b: list[Iterator[bool]] = []
            for a in _group_a:
                _results = executor.map(
                    partial(compare, compared=a, alg=fn, target=target, threshold=threshold),
                    _group_b
                )
                result_b.append(_results)
            results.append(result_b)
        return [any(map(any, result_b)) for result_b in results]


def _compare_groups_plateau(
    group_a: Sequence[T],
    *groups_b: Sequence[T],
    key: Optional[Callable[[T], str]] = None,
    alg: Optional[str] = None,
    target: str = 'pctArea',
    threshold: Optional[float] = None
) -> list[bool]:

    _group_a = group_a
    _groups_b = groups_b
    if key is not None:
        _group_a = map(key, _group_a)
        _groups_b = tuple(map(key, _group_b) for _group_b in _groups_b)

    if alg is None:
        alg = 'pixelwise'
    fn = ImageSorter.algs[alg]
    if threshold is None:
        threshold = ImageSorter.thresholds[alg]

    def _key(group_b: Sequence[T]) -> float:
        for a in _group_a:
            for b in group_b:
                if compare(a, b, fn, target, threshold):
                    return 1.
        return 0.

    def search_from_middle(
        a: Sequence[T], x: float, key: Callable[[T], float]
    ) -> int:
        'Locate ANY item greater than or equal to x'
        def recursive_search(start: int, end: int, depth: int = 0) -> int:
            if start > end:
                return -1  # Base case: no valid element found

            mid = (start + end) // 2

            # Check if the middle element meets the condition
            if key(a[mid]) >= x:
                return mid

            # Recursively search left and right simultaneously
            left_result = recursive_search(start, mid - 1, depth + 1)
            right_result = recursive_search(mid + 1, end, depth + 1)

            # Return the first valid element from either side
            if left_result != -1:
                return left_result
            return right_result

        return recursive_search(0, len(a) - 1)

    def locate_ge(
        a: Sequence[T], x: float, key: Callable[[T], float]
    ) -> int:
        'Locate leftmost item greater than or equal to x'
        i = bisect.bisect_left(a, x, key=key)
        if i != len(a):
            return i
        raise ValueError

    temp = [False for _ in _groups_b]
    mid = left = right = search_from_middle(_groups_b, 0.5, _key)
    if mid != -1:
        try:
            left = locate_ge(_groups_b[:mid], 0.5, _key)
        except ValueError:
            ...
        try:
            right += locate_ge(_groups_b[mid+1:], -0.5, lambda x: -_key(x))
        except ValueError:
            ...
        for i in range(left, right + 1):
            temp[i] = True
    return temp


def compare_groups(
    group_a: Sequence[T],
    *groups_b: Sequence[T],
    key: Optional[Callable[[T], str]] = None,
    alg: Optional[str] = None,
    target: str = 'pctArea',
    threshold: Optional[float] = None,
    kind: Literal['bruteforce', 'plateau'] = 'bruteforce'
) -> list[bool]:
    fn = _compare_groups_bruteforce if kind == 'bruteforce' else _compare_groups_plateau
    return fn(group_a, *groups_b, key=key, alg=alg, target=target, threshold=threshold)


class ImageIndex(TypedDict):
    i: int
    path: str | Path
    primary: Optional[int]
    secondary: Optional[int]


class ImageSorter:
    algs: dict[str, Callable[[str, str], DiffResult]] = {
        'pixelwise': cache(pixelwise), 'SSIM': cache(SSIM), 'ccip': ccip, 'lpips': cache(lpips)
    }
    thresholds: dict[str, float] = {
        'pixelwise': 0.1, 'SSIM': 0.8, 'ccip': 0., 'lpips': 0.55
    }

    def __init__(
        self,
        images: Iterable[str | Path],
        alg: str = 'pixelwise',
        threshold: Optional[float] = None,
        kind: str = 'primary',
        chunk: Optional[int] = None
    ):
        self.images: list[ImageIndex] = [
            ImageIndex(i=i, path=im, primary=None, secondary=None) for i, im in enumerate(images)
        ]
        self.threshold = threshold
        self._kind = kind
        self.chunk = chunk
        self.threshold = threshold if threshold is not None else self.thresholds[alg]
        self.alg = partial(self.algs[alg], cutoff=None)
        self._classify()

    def _classify(self) -> None:
        is_first = True
        chunk = self.chunk
        logger.info('Image classification start!')
        logger.info('Images are compared with {} chunk!'.format('{} per'.format(chunk) if chunk is not None else 'no'))
        while True:
            if is_first:
                remaining_images = self.images
                is_first = False
            else:
                remaining_images = list(filter(lambda x: x['primary'] is None, remaining_images))
            if (len(remaining_images) == 0):
                break
            remaining_images_in_chunk = remaining_images[:chunk]
            primary = remaining_images_in_chunk[0]['i']
            logger.info(
                'comparing "{}" with the remaining {} images in current chunk'.format(
                    Path(remaining_images_in_chunk[0]['path']).relative_to('.'),
                    len(remaining_images_in_chunk) - 1
                )
            )
            similar_images = get_grouped(
                *remaining_images_in_chunk,
                alg=self.alg,
                threshold=self.threshold,
                key=lambda x: x['path']
            )
            logger.info(
                '{} images similar to "{}" are found.'.format(
                    len(similar_images) - 1,
                    Path(remaining_images_in_chunk[0]['path']).relative_to('.')
                )
            )

            next_i = cur_secondary = primary
            for sim in similar_images:
                sim['primary'] = primary
                if sim['i'] not in (next_i + 1, next_i):
                    cur_secondary = sim['i']
                sim['secondary'] = cur_secondary
                next_i = sim['i']
                logger.debug(sim)

    @property
    def kind(self) -> str:
        if self._kind is None:
            return 'primary'
        else:
            return self._kind

    def classified(
            self, kind: Optional[str] = None, ret_key: Callable[[ImageIndex], T] = lambda im: im['path']
    ) -> list[list[T]]:
        if kind is None:
            kind = self.kind

        def key(x: ImageIndex):
            return x[kind]
        if kind == 'primary':
            images = sorted(self.images, key=key)
        else:
            images = self.images
        groups = [tuple(g) for _, g in groupby(images, key=key)]
        return [list(map(ret_key, g)) for g in groups]


def image_sorted(
        images: Iterable[str | Path],
        alg: str = 'pixelwise',
        threshold: Optional[float] = None,
        kind: str = 'primary',
        ret_key: Callable[[ImageIndex], T] = lambda im: im['path'],
        chunk: Optional[int] = None
) -> list[list[T]]:
    image_sorter = ImageSorter(
        get_all_images(list(map(Path, images))),
        alg=alg,
        threshold=threshold,
        kind=kind,
        chunk=chunk
    )
    return image_sorter.classified(ret_key=ret_key)


if __name__ == '__main__':
    import argparse
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs='+', type=Path, help="input image paths")
    ap.add_argument("--alg", '-A', type=str, default='pixelwise', help="comparasion algorithm")
    ap.add_argument("--threshold", '-T', type=float, default=0.4, help="threshold")
    ap.add_argument("--kind", '-K', type=str, default='primary',
                    help="classify kind", choices=['primary', 'secondary'])
    args = ap.parse_args()

    is_first = True
    for g in image_sorted(args.path, args.alg, args.threshold, args.kind):
        if is_first:
            is_first = False
        else:
            print()
        for f in g:
            print(f)
