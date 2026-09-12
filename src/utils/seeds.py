"""Single reproducible seed for the whole project."""

RANDOM_STATE = 42


def set_global_seed(seed: int = RANDOM_STATE) -> int:
    """Seed Python's ``random`` module and NumPy.

    Training libraries added in later phases should also use ``RANDOM_STATE``.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    return seed
