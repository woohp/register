import math
from random import Random

from .. import ext


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
    return [i + r for r, i in enumerate(ext.sord(k, sk))]
