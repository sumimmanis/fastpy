import clib


def cumsum(array):
    """A faster implementation of np.cumsum"""
    return clib.cumsum(array)
