from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_distance_squared[FloatingNDArray: (NDArray[np.float32], NDArray[np.float64])](
    x: FloatingNDArray, y: FloatingNDArray
) -> FloatingNDArray:
    diff = x - y
    return np.einsum("...i,...i->...", diff, diff)
