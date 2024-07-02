from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_distance_squared[FloatingNDArray: (NDArray[np.float32], NDArray[np.float64])](
    x: FloatingNDArray, y: FloatingNDArray
) -> FloatingNDArray:
    diff = x - y
    return np.einsum("...i,...i->...", diff, diff)


def get_pairwise_distances_squared[FloatingNDArray: (NDArray[np.float32], NDArray[np.float64])](
    A: FloatingNDArray, B: FloatingNDArray
) -> FloatingNDArray:
    # Compute squared Euclidean distances more efficiently
    a2 = (A[..., None, :] @ A[..., None]).squeeze(-1)
    b2 = (B[..., None, :] @ B[..., None]).squeeze(-1).squeeze(-1)[..., None, :]

    # Use broadcasting to compute the distance matrix
    D = a2 + b2 - 2 * (A @ B.swapaxes(-1, -2))

    return D
