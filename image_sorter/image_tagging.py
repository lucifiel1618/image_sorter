import functools
from typing import Callable, Iterable


_cached_taggers: dict[frozenset[str | None], Callable[[str], set[str]]] = {}


def get_image_tagger(models: Iterable[str | None]) -> Callable[[str], set[str]]:
    model_set = frozenset(models)

    try:
        return _cached_taggers[model_set]
    except KeyError:
        ...

    import imgutils.tagging

    tagger_fns = [
        functools.partial(
            getattr(imgutils.tagging, f'get_{model if model is not None else "wd14"}_tags'),
            fmt=('general',),
            general_mcut_enabled=True
        ) for model in model_set
    ]

    def fn(img: str) -> set[str]:
        return set.union(*(set(tagger_fn(img)[0]) for tagger_fn in tagger_fns))

    _cached_taggers[model_set] = fn

    return fn
