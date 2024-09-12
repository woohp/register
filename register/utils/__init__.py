import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(
    ["f4[::1](f4[:, ::1], f4[:, ::1])", "f4[:, ::1](f4[:, :, :], f4[:, :, :])"],
    cache=True,
    fastmath=True,
    boundscheck=False,
    parallel=True,
)
def get_distance_squared[FloatingNDArray: (NDArray[np.float32], NDArray[np.float64])](
    x: FloatingNDArray, y: FloatingNDArray
) -> FloatingNDArray:
    """Computes the _squared_ L2 distance between each pair of the two collections of row vectors.
    Roughly analogous to torch.cdist(x, y)**2.
    Unlike torch.cdist, this function also supports batched inputs of shape [N, M, D], where
    N is the batch size, M is the number of rows, and D is the number of columns.
    """
    dim = x.shape[-1]

    if x.ndim == 2:
        distances = np.empty((x.shape[0],), dtype=x.dtype)
        for i in numba.prange(x.shape[0]):
            d = 0.0
            for j in numba.prange(dim):
                d += (x[i, j] - y[i, j]) ** 2
            distances[i] = d

    else:
        distances = np.empty((x.shape[0], x.shape[1]), dtype=x.dtype)
        for i in numba.prange(x.shape[0]):
            for j in numba.prange(x.shape[1]):
                d = 0.0
                for k in numba.prange(dim):
                    d += (x[i, j, k] - y[i, j, k]) ** 2
                distances[i, j] = d

    return distances


@numba.njit(
    ["f4[:, ::1](f4[:, :], f4[:, :])", "f4[:, :, ::1](f4[:, :, :], f4[:, :, :])"],
    cache=True,
    fastmath=True,
    boundscheck=False,
    parallel=True,
)
def get_pairwise_distances_squared[FloatingNDArray: (NDArray[np.float32], NDArray[np.float64])](
    A: FloatingNDArray, B: FloatingNDArray
) -> FloatingNDArray:
    """Computes the _squared_ L2 distance between row in collection A and each row in collection B.
    Supports both batched and non-batched inputs.
    For non-batched inputs, the input A has shape [K1, D] and B has shape [K2, D], and the output has has shape
    [K1, K2]. For batched inputs, input A has shape [B, K1, D], B has shape [B, K2, D], and the output has shape
    [B, K1, K2].
    """

    dim = A.shape[-1]

    if A.ndim == 2:
        distances = np.empty((A.shape[0], B.shape[0]), dtype=A.dtype)
        for i in numba.prange(A.shape[0]):
            for j in numba.prange(B.shape[0]):
                d = 0.0
                for k in range(dim):
                    d += (A[i, k] - B[j, k]) ** 2
                distances[i, j] = d

    else:
        distances = np.empty((A.shape[0], A.shape[1], B.shape[1]), dtype=A.dtype)
        for b in numba.prange(A.shape[0]):
            for i in numba.prange(A.shape[1]):
                for j in numba.prange(B.shape[1]):
                    d = 0.0
                    for k in range(dim):
                        d += (A[b, i, k] - B[b, j, k]) ** 2
                    distances[b, i, j] = d

    return distances
