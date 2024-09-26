from collections.abc import Callable
from functools import partial
from typing import Any

import numba
import numpy as np
from numba import types
from numba.cpython import mathimpl
from numba.extending import intrinsic
from numpy.typing import NDArray


def numba_njit[F](*args: Any, **kwargs: Any) -> Callable[[F], F]:
    return numba.njit(*args, **kwargs)


nb_f32array = partial(types.Array, types.float32)


@numba_njit(
    [
        nb_f32array(1, "C")(nb_f32array(2, "C", readonly=True), nb_f32array(2, "C", readonly=True)),
        nb_f32array(2, "C")(nb_f32array(3, "C", readonly=True), nb_f32array(3, "C", readonly=True)),
        nb_f32array(2, "C")(nb_f32array(3, "C", readonly=True), nb_f32array(2, "C", readonly=True)),
    ],
    cache=True,
    fastmath=True,
    boundscheck=False,
    parallel=True,
)
def get_distance_squared(x: NDArray[np.float32], y: NDArray[np.float32]) -> NDArray[np.float32]:
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

        if y.ndim == 2:
            for i in numba.prange(x.shape[0]):
                for j in numba.prange(x.shape[1]):
                    d = 0.0
                    for k in numba.prange(dim):
                        d += (x[i, j, k] - y[j, k]) ** 2
                    distances[i, j] = d

        elif y.ndim == 3:
            distances = np.empty((x.shape[0], x.shape[1]), dtype=x.dtype)
            for i in numba.prange(x.shape[0]):
                for j in numba.prange(x.shape[1]):
                    d = 0.0
                    for k in numba.prange(dim):
                        d += (x[i, j, k] - y[i, j, k]) ** 2
                    distances[i, j] = d

    return distances


@numba_njit(
    [
        nb_f32array(2, "C")(nb_f32array(2, "A", readonly=True), nb_f32array(2, "A", readonly=True)),
        nb_f32array(3, "C")(nb_f32array(3, "A", readonly=True), nb_f32array(3, "A", readonly=True)),
    ],
    cache=True,
    fastmath=True,
    boundscheck=False,
    parallel=True,
)
def get_pairwise_distances_squared(A: NDArray[np.float32], B: NDArray[np.float32]) -> NDArray[np.float32]:
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


@intrinsic
def popcnt(typingctx, src):  # type: ignore[no-untyped-def]
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        return mathimpl.call_fp_intrinsic(builder, "llvm.ctpop.i64", args)

    return sig, codegen


@numba_njit("u1(u8)", inline="always", cache=True)
def popcount(x: int) -> int:
    return popcnt(x)


@numba_njit("u2[:, ::1](u8[:, ::1], u8[:, ::1])", fastmath=True, boundscheck=False, parallel=True, cache=True)
def hamming_distance_matrix(X: NDArray[np.uint64], Y: NDArray[np.uint64]) -> NDArray[np.uint16]:
    result = np.empty((X.shape[0], Y.shape[0]), dtype=np.uint16)

    for i in numba.prange(X.shape[0]):
        for j in numba.prange(Y.shape[0]):
            result[i, j] = (
                popcount(X[i, 0] ^ Y[j, 0])
                + popcount(X[i, 1] ^ Y[j, 1])
                + popcount(X[i, 2] ^ Y[j, 2])
                + popcount(X[i, 3] ^ Y[j, 3])
                + popcount(X[i, 4] ^ Y[j, 4])
                + popcount(X[i, 5] ^ Y[j, 5])
                + popcount(X[i, 6] ^ Y[j, 6])
                + popcount(X[i, 7] ^ Y[j, 7])
            )

    return result


@numba_njit("f4[:, :, ::1](i4, i4, f4[:, ::1])", cache=True, fastmath=True, boundscheck=False, parallel=True)
def generate_quadratic_remap_maps(height: int, width: int, T: NDArray[np.float32]) -> NDArray[np.float32]:
    result = np.empty((2, height, width), dtype=np.float32)

    for y in numba.prange(height):
        yy = y * y

        for x in numba.prange(width):
            xx = x * x
            xy = x * y

            result[0, y, x] = T[0, 0] + T[1, 0] * x + T[2, 0] * y + T[3, 0] * xx + T[4, 0] * xy + T[5, 0] * yy
            result[1, y, x] = T[0, 1] + T[1, 1] * x + T[2, 1] * y + T[3, 1] * xx + T[4, 1] * xy + T[5, 1] * yy

    return result
