import math
from random import Random

from . import numba_njit


def get_random_samples(N: int, k: int, max_samples: int, rng: Random) -> list[list[int]]:
    n_combos = math.comb(N, k)
    n_samples = min(n_combos, max_samples)

    out: list[list[int]] = []
    for i in range(n_samples):
        if n_combos < max_samples:
            idx = i
        else:
            idx = rng.randint(0, math.comb(min(k + i, N), k) - 1)

        out.append(xord(k, idx))

    return out


def xord(k: int, sk: int) -> list[int]:
    """
    Inverse function of ``ordx``, given output tuple size and compressed index.
    """
    return [i + r for r, i in enumerate(sord(k, sk))]


@numba_njit("List(i8)(u8, u8)", cache=True)
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
