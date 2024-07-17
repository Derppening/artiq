from typing import TypeVar, Generic, Union

import numpy as np
import numpy.typing as npt

__all__ = ["ndarray", "np_array", "int64", "np_sqrt"]


T = TypeVar('T')
N = TypeVar("N", bound=np.uint64)
class _NDArrayDummy(Generic[T, N]):
    pass

ndarray = Union[npt.NDArray[T], _NDArrayDummy[T, N]]


def np_array(object: npt.ArrayLike, copy=True, ndmin=0):
    return np.array(object, copy=copy, ndmin=ndmin)


def int64(x: npt.ArrayLike):
    return np.int64(x)


def np_sqrt(x: npt.ArrayLike):
    return np.sqrt(x)
