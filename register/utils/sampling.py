import math
from itertools import combinations
from random import Random

import numpy as np
from numpy.typing import NDArray

# Once the prefix saturates at N, dense rejection sampling wastes time on collisions.
# For small enough total combination spaces, materializing and shuffling the unseen
# suffix is faster once we need a meaningful fraction of what remains.
DENSE_SUFFIX_MAX_COMBOS = 100_000
DENSE_SUFFIX_MIN_FRACTION = 0.25


def get_random_samples(N: int, k: int, max_samples: int, rng: Random) -> NDArray[np.int64]:
    if k > N:
        raise ValueError(f"Cannot sample {k} items from a population of {N}")

    n_combos = math.comb(N, k)
    n_samples = min(n_combos, max_samples)

    if n_combos <= max_samples:
        # Caller asked for at least the full space, so return every combination.
        return np.array(list(combinations(range(N), k)), dtype=np.int64)

    seen: set[tuple[int, ...]] = set()

    for i in range(n_samples):
        # Preserve the original prefix-growth behavior: at iteration i we only draw
        # from combinations of the first min(k + i, N) points.
        prefix_size = min(k + i, N)

        if prefix_size == N:
            remaining_samples = n_samples - i
            remaining_combos = n_combos - len(seen)
            if (
                n_combos <= DENSE_SUFFIX_MAX_COMBOS
                and remaining_samples >= DENSE_SUFFIX_MIN_FRACTION * remaining_combos
            ):
                # Once the prefix stops growing, every remaining draw comes from the
                # same full combination space. To avoid too many collisions in the dense
                # tail, it can be cheaper to materialize the remaining unseen combinations
                # and keep what we need instead of continuing rejection sampling.
                all_combos = list(combinations(range(N), k))
                unseen = [combo for combo in all_combos if combo not in seen]
                rng.shuffle(unseen)
                seen.update(unseen[:remaining_samples])
                break

        while True:
            # Sample a canonical sorted subset so tuple equality matches set equality.
            sample = tuple(sorted(rng.sample(range(prefix_size), k)))
            if sample not in seen:
                seen.add(sample)
                break

    return np.array(list(seen), dtype=np.int64)
