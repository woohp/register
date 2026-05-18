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
def generate_bernstein2_remap_maps(height: int, width: int, T: NDArray[np.float32]) -> NDArray[np.float32]:
    result = np.empty((2, height, width), dtype=np.float32)
    max_dim = float(max(height, width))

    for y in numba.prange(height):
        fy = float(y) / max_dim
        for x in numba.prange(width):
            fx = float(x) / max_dim

            one_minus_x = 1.0 - fx
            one_minus_y = 1.0 - fy

            b00 = one_minus_x * one_minus_x * one_minus_y * one_minus_y
            b10 = 2.0 * fx * one_minus_x * one_minus_y * one_minus_y
            b20 = fx * fx * one_minus_y * one_minus_y
            b01 = one_minus_x * one_minus_x * 2.0 * fy * one_minus_y
            b11 = 2.0 * fx * one_minus_x * 2.0 * fy * one_minus_y
            b21 = fx * fx * 2.0 * fy * one_minus_y
            b02 = one_minus_x * one_minus_x * fy * fy
            b12 = 2.0 * fx * one_minus_x * fy * fy
            b22 = fx * fx * fy * fy

            result[0, y, x] = (
                T[0, 0] * b00
                + T[1, 0] * b10
                + T[2, 0] * b20
                + T[3, 0] * b01
                + T[4, 0] * b11
                + T[5, 0] * b21
                + T[6, 0] * b02
                + T[7, 0] * b12
                + T[8, 0] * b22
            )
            result[1, y, x] = (
                T[0, 1] * b00
                + T[1, 1] * b10
                + T[2, 1] * b20
                + T[3, 1] * b01
                + T[4, 1] * b11
                + T[5, 1] * b21
                + T[6, 1] * b02
                + T[7, 1] * b12
                + T[8, 1] * b22
            )

    return result


@numba_njit("f4[:, :, ::1](i4, i4, f4[:, ::1])", cache=True, fastmath=True, boundscheck=False, parallel=True)
def generate_tps_remap_maps(height: int, width: int, T: NDArray[np.float32]) -> NDArray[np.float32]:
    # T: [N+3, 4]
    # first 2 cols: weights and affine coeffs
    # last 2 cols: control points (first N rows)
    num_points = T.shape[0] - 3
    W = T[:num_points, :2]
    ab = T[num_points:, :2]
    P = T[:num_points, 2:]

    result = np.empty((2, height, width), dtype=np.float32)

    for y in numba.prange(height):
        # normalize y to [0, 1] relative to template max size ???
        # Actually T was estimated on normalized coordinates [0, 1]
        # So we need to normalize pixel coordinates
        max_dim = float(max(height, width))
        fy = float(y) / max_dim

        for x in numba.prange(width):
            fx = float(x) / max_dim

            # 1. Base affine part
            out_x = ab[0, 0] + ab[1, 0] * fx + ab[2, 0] * fy
            out_y = ab[0, 1] + ab[1, 1] * fx + ab[2, 1] * fy

            # 2. Kernel part
            for i in range(num_points):
                dx = fx - P[i, 0]
                dy = fy - P[i, 1]
                r2 = dx * dx + dy * dy
                if r2 > 1e-10:
                    u = r2 * np.log(r2)
                    out_x += W[i, 0] * u
                    out_y += W[i, 1] * u

            result[0, y, x] = out_x
            result[1, y, x] = out_y

    return result
