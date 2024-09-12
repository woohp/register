import numba
import numpy as np
from numba import types
from numba.cpython import mathimpl
from numba.extending import intrinsic
from numpy.typing import NDArray


@intrinsic
def popcnt(typingctx, src):  # type: ignore[no-untyped-def]
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        return mathimpl.call_fp_intrinsic(builder, "llvm.ctpop.i64", args)

    return sig, codegen


@numba.njit("u1(u8)", inline="always", cache=True)
def popcount(x: int) -> int:
    return popcnt(x)


@numba.njit("u2[:, ::1](u8[:, ::1], u8[:, ::1])", fastmath=True, boundscheck=False, parallel=True, cache=True)
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


@numba.njit("f4[:, :, ::1](i4, i4, f4[:, ::1])", cache=True, fastmath=True, boundscheck=False, parallel=True)
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


@numba.njit("List(i8)(u8, u8)", cache=True)
def sord(k: int, sk: int) -> list[int]:
    """
    Inverse function of ``ords``, given output tuple size
    and compressed index.
    """
    # implementation copied from here:
    # https://math.stackexchange.com/a/1370461

    # Allocate output array. Content does not matter here, only length.
    idx = [0] * k
    if k == 0:
        return idx

    s = sk
    if k > 1:
        # Find j such that binomial(j+k-1,k) <= s < binomial(j+k,k)
        j = 0
        prev_b = 0
        b = 1
        while b <= s:
            # Maintain invariants:
            # prev_b == binomial(j+k-1,k), b == binomial(j+k,k)
            prev_b = b
            j += 1
            b *= j + k
            b //= j
        b = prev_b
        # j is now the greatest index occurring in the tuple.
        # From now on, we will search backwards, decreasing j.
        for r in range(k - 1, 1, -1):
            # b == binomial(j+r,r+1) <= s < binomial(j+r+1,r+1)
            idx[r] = j
            s -= b
            # Update to b = binomial(j+r-1,r)
            b *= r + 1
            b //= j + r
            # Find j such that binomial(j+r-1,r) <= s < binomial(j+r,r)
            while b > s:
                j -= 1
                b *= j
                b //= j + r
        # Simplified handling of r = 1
        # b == binomial(j+r,r+1) <= s < binomial(j+r+1,r+1)
        idx[1] = j
        s -= b

    # Simplified handling of r = 0
    # binomial(j+r,r+1) == s iff j == s
    idx[0] = s
    return idx
