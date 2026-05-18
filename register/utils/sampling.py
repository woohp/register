import math
from random import Random

import numpy as np
from numpy.typing import NDArray

from . import numba_njit


def get_random_samples(N: int, k: int, max_samples: int, rng: Random) -> NDArray[np.int64]:
    n_combos = math.comb(N, k)
    n_samples = min(n_combos, max_samples)

    out = np.empty((n_samples, k), dtype=np.int64)
    for i in range(n_samples):
        if n_combos < max_samples:
            idx = i
        else:
            idx = rng.randint(0, math.comb(min(k + i, N), k) - 1)

        out[i] = xord(k, idx)

    return out


def xord(k: int, sk: int) -> list[int]:
    """
    Inverse function of ``ordx``, given output tuple size and compressed index.
    """
    return [i + r for r, i in enumerate(sord(k, sk))]


@numba_njit("i8(i8, i8)", cache=True)
def _binomial_safe(n: int, k: int) -> int:
    """Compute C(n,k) safely, returning INT64_MAX on overflow."""
    INT64_MAX = 9223372036854775807
    if k > n - k:
        k = n - k
    if k < 0:
        return 0
    result = 1
    for i in range(k):
        if result > INT64_MAX // (n - i):
            return INT64_MAX
        result = result * (n - i) // (i + 1)
    return result


@numba_njit("List(i8)(i8, i8)", cache=True)
def sord(k: int, sk: int) -> list[int]:
    """
    Inverse function of ``ords``, given output tuple size
    and compressed index.
    """
    idx = [0] * k
    if k == 0:
        return idx

    s = sk
    if k > 1:
        # Binary search for largest j such that C(j+k, k) <= s
        lo, hi = 0, 1
        while _binomial_safe(hi + k, k) <= s:
            hi *= 2
        j = -1
        while lo < hi:
            mid = (lo + hi) // 2
            if _binomial_safe(mid + k, k) <= s:
                j = mid
                lo = mid + 1
            else:
                hi = mid

        # After search: C(j+k, k) <= s < C(j+k+1, k)
        # Original algorithm needs: j such that C(j+k, k) > s, with b = C(j+k-1, k)
        b = _binomial_safe(j + k, k)
        j = j + 1

        # Search backwards, decreasing j
        for r in range(k - 1, 1, -1):
            idx[r] = j
            s -= b
            b = b * (r + 1) // (j + r)
            while b > s and j > 0:
                j -= 1
                b = b * j // (j + r)
        idx[1] = j
        s -= b

    idx[0] = s
    return idx
