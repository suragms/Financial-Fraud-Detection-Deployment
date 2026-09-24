"""Models package."""

__all__ = [
    "TEST_SIZE",
    "build_preprocessor",
    "build_pipeline",
    "make_stratified_split",
    "train_and_save",
]


def __getattr__(name: str):
    if name in __all__:
        from src.models import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
