import dataclasses
import multiprocessing.sharedctypes
from pathlib import Path
import multiprocessing
import multiprocessing.managers
from typing import Any, Callable, Iterator, Literal, Optional, TypeAlias

import numpy as np
from PIL import Image
import imageio


Features: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
MatLike: TypeAlias = np.ndarray[tuple[int, int, int, int], np.dtype[np.float64 | np.int_]]


def get_image(f: Path) -> Image.Image:
    return Image.fromarray(imageio.imread(str(f)))


@dataclasses.dataclass(frozen=True, slots=True)
class ClassificationContext:
    category: Literal['label', 'compare'] = 'compare'
    alg: str = 'pixelwise'
    threshold: Optional[float] = None

    def meta_str(self, e: 'Element') -> str | None:
        ...


class SharedNDArray:
    """
    A numpy array wrapper using RawArray for zero-copy sharing across processes.
    Supports multiple arrays stored in shared memory with thread/process-safe access.
    """
    __slots__ = ('shapes', 'dtype', 'sizes', '_buffer', 'lock', '_offsets', 'lock', '_ready_event')

    def __init__(self, shapes: tuple[tuple[int, ...], ...], dtype: np.dtype = np.dtype(np.float64)):
        """
        Initialize a SharedNDArray with given shapes and dtypes.

        Args:
            shapes: tuple of tuples, each defining dimensions for one array
            dtypes: tuple of numpy dtypes (one per array) or single dtype for all arrays
        """
        self.shapes = tuple(tuple(s) if not isinstance(s, tuple) else s for s in shapes)

        # Handle single dtype or tuple of dtypes
        self.dtype = np.dtype(dtype)

        self.sizes = tuple(int(np.prod(shape)) for shape in self.shapes)

        # Calculate byte offsets for each array in the shared buffer
        offsets = tuple()
        total_bytes = 0
        for size in self.sizes:
            offsets += (total_bytes,)
            total_bytes += size * self.dtype.itemsize
        self._offsets = offsets
        # Create single shared buffer to hold all arrays
        self._buffer = multiprocessing.RawArray('b', total_bytes)

        self.lock = multiprocessing.Lock()

        self._ready_event = multiprocessing.Event()

    def _as_ndarrays(self) -> Iterator[Features]:
        """
        Return numpy ndarray views of all shared memory arrays (zero-copy).

        Yields:
            np.ndarray: View of each shared array with correct shape and dtype
        """

        dtype = self.dtype
        for (shape, size, offset) in zip(self.shapes, self.sizes, self._offsets):
            # Create a view at the correct offset with the correct dtype
            yield np.frombuffer(self._buffer, dtype=dtype, count=size, offset=offset).reshape(shape)

    def as_ndarrays(self) -> Iterator[Features]:
        self._ready_event.wait()
        yield from self._as_ndarrays()

    def update_from_ndarrays(self, *arrays: Features):
        """
        Update the shared arrays from provided numpy arrays.

        Args:
            *arrays: numpy arrays to copy into shared memory (must match shapes and dtypes)
        """
        if len(arrays) != len(self.shapes):
            raise ValueError(f"Expected {len(self.shapes)} arrays, got {len(arrays)}")

        shared_arrays = self._as_ndarrays()
        for (src, dst) in zip(arrays, shared_arrays):
            src = np.ascontiguousarray(src)
            np.copyto(dst=dst, src=src.ravel())

    def update_from_worker(self, fn: Callable[[], tuple[Features, ...]]):
        """
        Update the shared arrays by calling a worker function that returns numpy arrays.

        Args:
            fn: Callable that returns a tuple of numpy arrays to copy into shared memory
        """
        if self._ready_event.is_set():
            return
        with self.lock:
            if not self._ready_event.is_set():  # Double-check after locked is necessary to avoid race
                self.update_from_ndarrays(*fn())
                self._ready_event.set()

    @classmethod
    def from_ndarrays(cls, *arrays: Features):
        """
        Create a SharedNDArray from existing numpy arrays.

        Args:
            *arrays: numpy arrays to convert

        Returns:
            SharedNDArray: New instance containing the array data
        """
        shapes = tuple(arr.shape for arr in arrays)
        dtype = arrays[0].dtype

        assert all(dtype == dt for dt in arrays[1:]), "All arrays must have the same dtype"

        shared = cls(shapes, dtype)
        shared.update_from_ndarrays(*arrays)

        return shared

    def is_ready(self) -> bool:
        """
        Check if arrays have been written.

        Returns:
            bool: True if arrays have been written, False otherwise
        """
        return self._ready_event.is_set()


class NDArrayManager(multiprocessing.managers.BaseManager):
    ...


NDArrayManager.register('SharedNDArray', SharedNDArray)


@dataclasses.dataclass(frozen=True, slots=True)
class Element:
    path: Path
    _: dataclasses.KW_ONLY
    _meta: dict[ClassificationContext, dict[str, Any]] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    def set(self, sorter: ClassificationContext, **meta: Any):
        self._meta.setdefault(sorter, {}).update(meta)

    def get(self, sorter: ClassificationContext, key: str, default_value=None) -> Any:
        try:
            return self._meta[sorter][key]
        except KeyError:
            return default_value

    def get_image(self) -> Image.Image:
        return get_image(self.path)

    @classmethod
    def as_element(cls, inputlike: 'InputLike') -> 'Element':
        if isinstance(inputlike, (str, Path)):
            return cls(Path(inputlike))
        return inputlike

    def as_strdict(self) -> dict[str, str]:
        meta_str = '; '.join(meta for sorter in self._meta if (meta := sorter.meta_str(self)) is not None)
        return {'path': str(self.path), 'meta': f' # \\\\ {meta_str}' if meta_str else ''}

    def as_str(self, path_extra: Optional[str] = None) -> str:
        if path_extra is None:
            path_extra = ''
        return '{path}{path_extra}{meta}'.format(path_extra=path_extra, **self.as_strdict())


ElementGroup: TypeAlias = list[Element]
ElementGroups: TypeAlias = list[ElementGroup]
InputLike: TypeAlias = Element | str | Path
