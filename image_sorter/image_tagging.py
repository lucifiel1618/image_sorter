import functools
from typing import Callable, Iterable
from .element import Element, InputLike, ElementGroup, ElementGroups, ClassificationContext


class ImageTagger:
    _cached_taggers: dict[frozenset[str | None], Callable[[str], set[str]]] = {}

    def __init__(self, models: Iterable[str | None], context: ClassificationContext | None = None):
        if context is None:
            context = self.__class__.get_context(models)
        assert context.category == 'label', "ClassificationContext category for ImageTagger must be 'label'"
        self.context = context
        self.tagger_fn: Callable[[str], set[str]] = self.__class__.get_image_tagger(models)

    @classmethod
    def get_model_name(cls, model: str | None) -> str:
        return model if model is not None else 'wd14'

    @classmethod
    def get_context(cls, models: Iterable[str | None]) -> ClassificationContext:
        return ClassificationContext(
            'label',
            ','.join(sorted(frozenset(map(cls.get_model_name, models))))
        )

    @classmethod
    def get_image_tagger(cls, models: Iterable[str | None]) -> Callable[[str], set[str]]:
        model_set = frozenset(map(cls.get_model_name, models))

        try:
            return cls._cached_taggers[model_set]
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

        cls._cached_taggers[model_set] = fn

        return fn

    def _get_image_tags(self, img_path: str) -> set[str]:
        return self.tagger_fn(img_path)

    def get_element_tags(self, element: Element) -> set[str]:
        tags = element.get(self.context, 'tags')
        if tags is not None:
            return tags
        tags = self._get_image_tags(str(element.path))
        element.set(self.context, tags=tags)
        return tags

    def __call__(self, element: Element) -> set[str]:
        return self.get_element_tags(element)


def get_image_tagger(models: Iterable[str | None], context: ClassificationContext | None = None) -> ImageTagger:
    return ImageTagger(models, context)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Tag images using specified models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('images', nargs='+', help='Paths to image files to be tagged')
    parser.add_argument('--models', nargs='*', default=['wd14'], help='Models to use for tagging')
    args = parser.parse_args()
    tagger = get_image_tagger(args.models)
    for img_path in args.images:
        tags = tagger._get_image_tags(img_path)
        print(f'Image: {img_path}\nTags: {", ".join(sorted(tags))}\n')


if __name__ == '__main__':
    main()
