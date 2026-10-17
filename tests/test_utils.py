from collections.abc import Iterable, MutableSequence, Sequence
from random import Random
from typing import Any

import numpy as np
import pytest

from register.utils.sampling import get_random_samples


def test_get_random_samples_rejects_too_many_items() -> None:
    with pytest.raises(ValueError, match="Cannot sample"):
        get_random_samples(2, 3, 10, Random(4321))


def test_get_random_samples_returns_all_combinations_when_below_limit() -> None:
    output = get_random_samples(5, 3, 20, Random(0))
    expected = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [0, 2, 3],
            [0, 2, 4],
            [0, 3, 4],
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(output, expected)


def test_get_random_samples_returns_unique_sorted_subsets() -> None:
    output = get_random_samples(10, 3, 100, Random(1234))
    output_tuples = [tuple(row) for row in output.tolist()]

    assert len(output_tuples) == len(set(output_tuples))
    assert all(list(row) == sorted(row) for row in output_tuples)


class CountingRandom(Random):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.sample_calls = 0
        self.shuffle_calls = 0

    def sample[T](
        self, population: Sequence[T], k: int, *, counts: Iterable[int] | None = None
    ) -> list[T]:
        self.sample_calls += 1
        return super().sample(population, k, counts=counts)

    def shuffle(self, x: MutableSequence[Any]) -> None:
        self.shuffle_calls += 1
        super().shuffle(x)


def test_get_random_samples_keeps_unique_sorted_subsets_for_large_spaces() -> None:
    rng = CountingRandom(1234)
    output = get_random_samples(5_000, 6, 4_333, rng)
    output_tuples = [tuple(row) for row in output.tolist()]

    assert rng.sample_calls > 0
    assert len(output_tuples) == len(set(output_tuples))
    assert all(list(row) == sorted(row) for row in output_tuples)
    assert all(0 <= value < 5_000 for row in output_tuples for value in row)


def test_get_random_samples_materializes_suffix_when_prefix_saturates() -> None:
    rng = CountingRandom(1234)
    output = get_random_samples(30, 3, 4_000, rng)
    output_tuples = [tuple(row) for row in output.tolist()]

    assert output.shape == (4_000, 3)
    assert rng.shuffle_calls == 1
    assert rng.sample_calls >= 27
    assert len(output_tuples) == len(set(output_tuples))
    assert all(list(row) == sorted(row) for row in output_tuples)
    assert all(0 <= value < 30 for row in output_tuples for value in row)
