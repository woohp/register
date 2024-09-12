import numpy as np

from register.utils import get_distance_squared, get_pairwise_distances_squared


def test_get_distance_squared() -> None:
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [9, 7]], dtype=np.float32)

    np.testing.assert_allclose(get_distance_squared(x, y), [32, 45])


def test_get_distance_squared_batched() -> None:
    x = np.random.normal(0, 1, (3, 5, 4)).astype(np.float32)
    y = np.random.normal(0, 1, (3, 5, 4)).astype(np.float32)

    expected = np.sum((x - y) ** 2, axis=-1)
    np.testing.assert_allclose(get_distance_squared(x, y), expected, atol=1e-5)


def test_get_pairwise_distances_squared() -> None:
    x = np.random.normal(0, 1, (5, 4)).astype(np.float32)
    y = np.random.normal(0, 1, (7, 4)).astype(np.float32)

    expected = np.sum((x[:, None] - y[None]) ** 2, axis=-1)
    output = get_pairwise_distances_squared(x, y)
    assert output.shape == (5, 7)
    np.testing.assert_allclose(output, expected, atol=1e-5)


def test_get_pairwise_distances_squared_batched() -> None:
    x = np.random.normal(0, 1, (3, 5, 4)).astype(np.float32)
    y = np.random.normal(0, 1, (3, 7, 4)).astype(np.float32)

    expected = np.sum((x[:, :, None] - y[:, None]) ** 2, axis=-1)
    output = get_pairwise_distances_squared(x, y)
    assert output.shape == (3, 5, 7)
    np.testing.assert_allclose(output, expected, atol=1e-5)
