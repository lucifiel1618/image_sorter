#!/opt/homebrew/bin/python3

import bisect
import concurrent.futures
from pathlib import Path
import mimetypes
from itertools import groupby
from functools import partial
from typing import Callable, Iterable, Iterator, Literal, Optional, Sequence, TypedDict
import easy_logging
from .element import Element, InputLike, ElementGroup, ElementGroups, ClassificationContext
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


def get_all_images_as_elements(path: Sequence[Path]) -> Iterator[Element]:
    yield from map(Element, get_all_images(path))


def assgin_meta(e: Element, meta) -> None:
    e._meta.clear()
    e._meta.update(meta)


def compare(
    original: Element,
    compared: Element,
    alg: Callable[[Element, Element], DiffResult],
    target: str,
    threshold: float
) -> bool:
    return (alg(original, compared)[target] > threshold)


def get_grouped_results(
    original: Element,
    *compared_group: Element,
    alg: Callable[[Element, Element], DiffResult] = pixelwise,
    threshold: float = 0.1,
    target: str = 'pctArea',
    executor: Optional[concurrent.futures.Executor] = None
) -> Iterator[tuple[bool, Element]]:

    yield (True, original)

    if len(compared_group) == 0:
        return

    if hasattr(alg, 'prepare_features'):
        for e in (original, *compared_group):
            alg.prepare_features(e)

    fn = partial(compare, compared=original, alg=alg, target=target, threshold=threshold)

    if executor is None:
        executor = concurrent.futures.ProcessPoolExecutor()
        owner = True
    else:
        owner = False

    g = compared_group
    yield from zip(executor.map(fn, g), g)

    if owner:
        executor.shutdown()


def get_grouped(
    original: Element,
    *compared_group: Element,
    alg: Callable[[Element, Element], DiffResult] = pixelwise,
    threshold: float = 0.1,
    target: str = 'pctArea',
    executor: Optional[concurrent.futures.Executor] = None
) -> ElementGroup:
    return ElementGroup(
        e for is_similar, e in get_grouped_results(
            original, *compared_group, alg=alg, threshold=threshold, target=target, executor=executor
        ) if is_similar
    )


def _compare_groups_bruteforce(
    group_a: ElementGroup,
    *groups_b: ElementGroup,
    alg: Optional[str] = None,
    target: str = 'pctArea',
    threshold: Optional[float] = None,
    executor: Optional[concurrent.futures.Executor] = None
) -> list[bool]:

    if alg is None:
        alg = 'pixelwise'
    alg_fn = ImageSorter.algs[alg]
    if threshold is None:
        threshold = ImageSorter.thresholds[alg]

    if executor is None:
        executor = concurrent.futures.ProcessPoolExecutor()
        owner = True
    else:
        owner = False

    results: list[list[Iterator[bool]]] = []
    for group_b in groups_b:
        result_b: list[Iterator[bool]] = []
        for a in group_a:
            grouped_results = get_grouped_results(
                a, *group_b, alg=alg_fn, threshold=threshold, target=target, executor=executor
            )
            result_b.append((is_similar for is_similar, _ in grouped_results))
        results.append(result_b)

    if owner:
        executor.shutdown()
    return [any(map(any, result_b)) for result_b in results]


def _compare_groups_plateau(
    group_a: ElementGroup,
    *groups_b: ElementGroup,
    alg: Optional[str] = None,
    target: str = 'pctArea',
    threshold: Optional[float] = None,
    executor: Optional[concurrent.futures.Executor] = None
) -> list[bool]:

    if alg is None:
        alg = 'pixelwise'
    fn = ImageSorter.algs[alg]
    if threshold is None:
        threshold = ImageSorter.thresholds[alg]

    def key(group_b: ElementGroup) -> float:
        for a in group_a:
            for b in group_b:
                c = compare(a, b, fn, target, threshold)
                if c:
                    return 1.
        return 0.

    def search_from_middle(
        a: tuple[ElementGroup, ...], x: float, key: Callable[[ElementGroup], float]
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
        a: tuple[ElementGroup, ...], x: float, key: Callable[[ElementGroup], float]
    ) -> int:
        'Locate leftmost item greater than or equal to x'
        i = bisect.bisect_left(a, x, key=key)
        if i != len(a):
            return i
        raise ValueError

    temp = [False for _ in groups_b]
    mid = left = right = search_from_middle(groups_b, 0.5, key)
    if mid != -1:
        try:
            left = locate_ge(groups_b[:mid], 0.5, key)
        except ValueError:
            ...
        try:
            right += locate_ge(groups_b[mid + 1:], -0.5, lambda x: -key(x))
        except ValueError:
            ...
        for i in range(left, right + 1):
            temp[i] = True
    return temp


def compare_groups(
    group_a: ElementGroup,
    *groups_b: ElementGroup,
    alg: Optional[str] = None,
    target: str = 'pctArea',
    threshold: Optional[float] = None,
    kind: Literal['bruteforce', 'plateau'] = 'bruteforce',
    executor: Optional[concurrent.futures.Executor] = None
) -> list[bool]:
    fn = _compare_groups_bruteforce if kind == 'bruteforce' else _compare_groups_plateau
    return fn(group_a, *groups_b, alg=alg, target=target, threshold=threshold, executor=executor)


class ImageIndex(TypedDict):
    i: int
    primary: Optional[int]
    secondary: Optional[int]


class ImageSorter:
    algs: dict[str, Callable[[InputLike, InputLike], DiffResult]] = {
        'pixelwise': pixelwise, 'SSIM': SSIM, 'ccip': ccip, 'lpips': lpips
    }
    thresholds: dict[str, float] = {
        'pixelwise': 0.1, 'SSIM': 0.8, 'ccip': 0., 'lpips': 0.55
    }

    def __init__(
        self,
        images: Iterable[Element],
        alg: str = 'pixelwise',
        threshold: Optional[float] = None,
        kind: str = 'primary',
        chunk: Optional[int] = None,
        target: str = 'pctArea'
    ):

        self.sorter_context = ClassificationContext(alg=alg, threshold=threshold)
        self.images = [
            im for i, im in enumerate(images)
            if im.set(self.sorter_context, index=ImageIndex(i=i, primary=None, secondary=None)) is None
        ]
        self._kind = kind
        self.chunk = chunk
        self.threshold = threshold if threshold is not None else self.thresholds[alg]
        self.alg = partial(self.algs[alg], cutoff=None)
        self.target = target
        self._classify()

    def _classify(self) -> None:
        remaining_images: ElementGroup
        is_first = True
        chunk = self.chunk
        logger.info('Image classification start!')
        logger.info('Images are compared with {} chunk!'.format('{} per'.format(chunk) if chunk is not None else 'no'))
        while True:
            if is_first:
                remaining_images = self.images
                is_first = False
            else:
                remaining_images = ElementGroup(
                    filter(
                        lambda e: e.get(self.sorter_context, 'primary') is None,
                        remaining_images
                    )
                )
            if (len(remaining_images) == 0):
                break
            remaining_images_in_chunk = remaining_images[:chunk]
            primary = remaining_images_in_chunk[0].get(self.sorter_context, 'index')['i']
            logger.info(
                'comparing "{}" with the remaining {} images in current chunk'.format(
                    remaining_images_in_chunk[0].path.relative_to('.'),
                    len(remaining_images_in_chunk) - 1
                )
            )
            similar_images = get_grouped(
                *remaining_images_in_chunk,
                alg=self.alg,
                threshold=self.threshold,
                target=self.target
            )
            logger.info(
                '{} images similar to "{}" are found.'.format(
                    len(similar_images) - 1,
                    remaining_images_in_chunk[0].path.relative_to('.')
                )
            )

            next_i = cur_secondary = primary
            for sim in similar_images:
                image_index = sim.get(self.sorter_context, 'index')
                image_index['primary'] = primary
                if image_index['i'] not in (next_i + 1, next_i):
                    cur_secondary = image_index['i']
                image_index['secondary'] = cur_secondary
                next_i = image_index['i']
                logger.debug(sim)

    @property
    def kind(self) -> str:
        if self._kind is None:
            return 'primary'
        else:
            return self._kind

    def classified(
            self,
            kind: Optional[str] = None
    ) -> ElementGroups:
        if kind is None:
            kind = self.kind

        def index_key(x: Element) -> int:
            index = x.get(self.sorter_context, 'index')
            return index[kind]

        if kind == 'primary':
            images = sorted(self.images, key=index_key)
        else:
            images = self.images
        groups = ElementGroups(ElementGroup(g) for _, g in groupby(images, key=index_key))
        return groups


def image_sorted[T](
        images: Iterable[T],
        alg: str = 'pixelwise',
        threshold: Optional[float] = None,
        kind: str = 'primary',
        key: Callable[[T], InputLike] = lambda im: str(im) if not isinstance(im, InputLike) else im,
        chunk: Optional[int] = None,
        target: str = 'pctArea'
) -> list[list[T]]:
    element_dict = {}
    for im in images:
        element_dict.setdefault(Element.as_element(key(im)), []).append(im)
    image_sorter = ImageSorter(
        element_dict.keys(),
        alg=alg,
        threshold=threshold,
        kind=kind,
        chunk=chunk,
        target=target
    )
    return [[element_dict[im].pop() for im in im_group] for im_group in image_sorter.classified()]


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
    for g in image_sorted(
        map(Element.as_element, args.path), args.alg, args.threshold, args.kind
    ):
        if is_first:
            is_first = False
        else:
            print()
        for f in g:
            print(f)
