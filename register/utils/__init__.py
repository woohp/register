from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import NBitBase, NDArray

FloatingNDArray = TypeVar("FloatingNDArray", NDArray[np.float32], NDArray[np.float64])


def get_distance_squared(x: FloatingNDArray, y: FloatingNDArray) -> FloatingNDArray:
    diff = x - y
    return np.einsum("...i,...i->...", diff, diff)
