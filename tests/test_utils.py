from random import Random

import numpy as np
import pytest
from numpy.typing import NDArray

from register.utils import get_distance_squared, get_pairwise_distances_squared
from register.utils.sampling import get_random_samples


@pytest.mark.parametrize(
    ("batch_size", "num_points", "dim"),
    [
        (None, 2, 2),
        (3, 5, 4),
    ],
)
def test_get_distance_squared(batch_size: int | None, num_points: int, dim: int) -> None:
    shape = (num_points, dim) if batch_size is None else (batch_size, num_points, dim)
    x = np.random.normal(0, 1, shape).astype(np.float32)
    y = np.random.normal(0, 1, shape).astype(np.float32)

    expected = np.sum((x - y) ** 2, axis=-1)
    np.testing.assert_allclose(get_distance_squared(x, y), expected, atol=1e-5)


@pytest.mark.parametrize(
    ("batch_size", "n", "m", "dim"),
    [
        (None, 5, 7, 4),
        (3, 5, 7, 4),
    ],
)
def test_get_pairwise_distances_squared(batch_size: int | None, n: int, m: int, dim: int) -> None:
    x_shape = (n, dim) if batch_size is None else (batch_size, n, dim)
    y_shape = (m, dim) if batch_size is None else (batch_size, m, dim)

    x = np.random.normal(0, 1, x_shape).astype(np.float32)
    y = np.random.normal(0, 1, y_shape).astype(np.float32)

    expected: NDArray[np.float32]
    expected_shape: tuple[int, ...]

    if batch_size is None:
        expected = np.sum((x[:, None] - y[None]) ** 2, axis=-1)
        expected_shape = (n, m)
    else:
        expected = np.sum((x[:, :, None] - y[:, None]) ** 2, axis=-1)
        expected_shape = (batch_size, n, m)

    output = get_pairwise_distances_squared(x, y)
    assert output.shape == expected_shape
    np.testing.assert_allclose(output, expected, atol=1e-5)


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

    def sample(self, population: range, k: int) -> list[int]:
        self.sample_calls += 1
        return super().sample(population, k)

    def shuffle(self, x: list[tuple[int, ...]]) -> None:
        self.shuffle_calls += 1
        super().shuffle(x)


def test_get_random_samples_keeps_unique_sorted_ordered_subsets_for_large_spaces() -> None:
    rng = CountingRandom(1234)
    output = get_random_samples(5_000, 6, 4_333, rng)
    output_tuples = [tuple(row) for row in output.tolist()]

    assert rng.sample_calls > 0
    assert len(output_tuples) == len(set(output_tuples))
    assert all(list(row) == sorted(row) for row in output_tuples)
    assert all(0 <= value < 5_000 for row in output_tuples for value in row)


def test_get_random_samples_switches_to_materialized_suffix_when_prefix_saturates() -> None:
    rng = CountingRandom(1234)
    output = get_random_samples(30, 3, 4_000, rng)
    output_tuples = [tuple(row) for row in output.tolist()]

    assert output.shape == (4_000, 3)
    assert rng.shuffle_calls == 1
    assert rng.sample_calls >= 27
    assert len(output_tuples) == len(set(output_tuples))
    assert all(list(row) == sorted(row) for row in output_tuples)
    assert all(0 <= value < 30 for row in output_tuples for value in row)
