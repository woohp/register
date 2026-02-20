import numpy as np
import pytest

from register.utils import get_distance_squared, get_pairwise_distances_squared


@pytest.mark.parametrize(
    "batch_size, num_points, dim",
    [
        (None, 2, 2),
        (3, 5, 4),
    ],
)
def test_get_distance_squared(batch_size, num_points, dim) -> None:
    shape = (num_points, dim) if batch_size is None else (batch_size, num_points, dim)
    x = np.random.normal(0, 1, shape).astype(np.float32)
    y = np.random.normal(0, 1, shape).astype(np.float32)

    expected = np.sum((x - y) ** 2, axis=-1)
    np.testing.assert_allclose(get_distance_squared(x, y), expected, atol=1e-5)


@pytest.mark.parametrize(
    "batch_size, n, m, dim",
    [
        (None, 5, 7, 4),
        (3, 5, 7, 4),
    ],
)
def test_get_pairwise_distances_squared(batch_size, n, m, dim) -> None:
    x_shape = (n, dim) if batch_size is None else (batch_size, n, dim)
    y_shape = (m, dim) if batch_size is None else (batch_size, m, dim)

    x = np.random.normal(0, 1, x_shape).astype(np.float32)
    y = np.random.normal(0, 1, y_shape).astype(np.float32)

    if batch_size is None:
        expected = np.sum((x[:, None] - y[None]) ** 2, axis=-1)
        expected_shape = (n, m)
    else:
        expected = np.sum((x[:, :, None] - y[:, None]) ** 2, axis=-1)
        expected_shape = (batch_size, n, m)

    output = get_pairwise_distances_squared(x, y)
    assert output.shape == expected_shape
    np.testing.assert_allclose(output, expected, atol=1e-5)
