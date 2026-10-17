from collections.abc import Callable
from typing import Any

import numba
import numpy as np
from numba import types
from numba.cpython import mathimpl
from numba.extending import intrinsic
from numpy.typing import NDArray


def numba_njit[F](*args: Any, **kwargs: Any) -> Callable[[F], F]:
    return numba.njit(*args, **kwargs)


@intrinsic
def popcnt(typingctx, src):  # type: ignore[no-untyped-def]
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
        return mathimpl.call_fp_intrinsic(builder, "llvm.ctpop.i64", args)

    return sig, codegen


@numba_njit("u1(u8)", inline="always", cache=True)
def popcount(x: int) -> int:
    return popcnt(x)  # type: ignore[call-arg]


@numba_njit(
    "u2[:, ::1](u8[:, ::1], u8[:, ::1])",
    fastmath=True,
    boundscheck=False,
    parallel=True,
    cache=True,
)
def hamming_distance_matrix(
    X: NDArray[np.uint64], Y: NDArray[np.uint64]
) -> NDArray[np.uint16]:
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
