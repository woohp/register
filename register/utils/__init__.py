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
    """Compute element-wise squared L2 distance between corresponding row vectors.

    For 2-D inputs (N, D): returns (N,) with dist[i] = ||x[i] - y[i]||².
    For 3-D + 2-D inputs (B, N, D) + (N, D): returns (B, N) with dist[i,j] = ||x[i,j] - y[j]||².
    For 3-D + 3-D inputs (B, N, D) + (B, N, D): returns (B, N) with dist[i,j] = ||x[i,j] - y[i,j]||².
    """
    dim = x.shape[-1]

    if x.ndim == 2:
        distances = np.empty((x.shape[0],), dtype=x.dtype)
        for i in numba.prange(x.shape[0]):
            d = 0.0
            for j in range(dim):
                d += (x[i, j] - y[i, j]) ** 2
            distances[i] = d

    else:
        distances = np.empty((x.shape[0], x.shape[1]), dtype=x.dtype)

        if y.ndim == 2:
            for i in numba.prange(x.shape[0]):
                for j in range(x.shape[1]):
                    d = 0.0
                    for k in range(dim):
                        d += (x[i, j, k] - y[j, k]) ** 2
                    distances[i, j] = d

        elif y.ndim == 3:
            for i in numba.prange(x.shape[0]):
                for j in range(x.shape[1]):
                    d = 0.0
                    for k in range(dim):
                        d += (x[i, j, k] - y[i, j, k]) ** 2
                    distances[i, j] = d

    return distances


@intrinsic
def popcnt(typingctx, src):  # type: ignore[no-untyped-def]
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        return mathimpl.call_fp_intrinsic(builder, "llvm.ctpop.i64", args)

    return sig, codegen


@numba_njit("u1(u8)", inline="always", cache=True)
def popcount(x: int) -> int:
    return popcnt(x)  # type: ignore[call-arg]


@numba_njit("u2[:, ::1](u8[:, ::1], u8[:, ::1])", fastmath=True, boundscheck=False, parallel=True, cache=True)
def hamming_distance_matrix(X: NDArray[np.uint64], Y: NDArray[np.uint64]) -> NDArray[np.uint16]:
    result = np.empty((X.shape[0], Y.shape[0]), dtype=np.uint16)

    for i in numba.prange(X.shape[0]):
        for j in range(Y.shape[0]):
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
