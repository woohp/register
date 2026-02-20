import numpy as np
import pytest

import register.transforms


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_rigid_transform(batch_size) -> None:
    if batch_size is None:
        theta = 1.234
        tx, ty = -3, 5
        T = np.array(
            [
                [tx, np.cos(theta), -np.sin(theta)],
                [ty, np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float32,
        ).T
        A = np.random.randn(91, 3).astype(np.float32)
        expected_shape = (3, 2)
    else:
        tx = np.random.uniform(-10, 10, (batch_size,))
        ty = np.random.uniform(-10, 10, (batch_size,))
        thetas = np.random.uniform(-np.pi, np.pi, (batch_size,))
        T = np.array(
            [
                [tx, np.cos(thetas), -np.sin(thetas)],
                [ty, np.sin(thetas), np.cos(thetas)],
            ],
            dtype=np.float32,
        ).transpose(2, 1, 0)
        A = np.random.randn(batch_size, 91, 3).astype(np.float32)
        expected_shape = (batch_size, 3, 2)

    A[..., 0] = 1
    Y = A @ T

    estimated_T = register.transforms.RigidTransform.estimate(A, Y, 0.0)
    assert estimated_T.shape == expected_shape
    np.testing.assert_array_almost_equal(estimated_T, T, 5)
    assert estimated_T.dtype == np.float32

    if batch_size is None:
        T_scaled = T.copy()
        T_scaled[1:] *= 2
        Y_scaled = A @ T_scaled
        estimated_T_scaled = register.transforms.RigidTransform.estimate(A, Y_scaled, 0.0)
        assert not np.allclose(estimated_T_scaled, T_scaled, atol=1e-5)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_similarity_transform(batch_size) -> None:
    if batch_size is None:
        theta = 1.234
        tx, ty = -3, 5
        scale = 0.123
        T = np.array(
            [
                [tx, np.cos(theta) * scale, -np.sin(theta) * scale],
                [ty, np.sin(theta) * scale, np.cos(theta) * scale],
            ],
            dtype=np.float32,
        ).T
        A = np.random.randn(91, 3).astype(np.float32)
        expected_shape = (3, 2)
    else:
        tx = np.random.uniform(-10, 10, (batch_size,))
        ty = np.random.uniform(-10, 10, (batch_size,))
        thetas = np.random.uniform(-np.pi, np.pi, (batch_size,))
        scales = np.random.uniform(-10, 10, (batch_size,))
        T = np.array(
            [
                [tx, np.cos(thetas) * scales, -np.sin(thetas) * scales],
                [ty, np.sin(thetas) * scales, np.cos(thetas) * scales],
            ],
            dtype=np.float32,
        ).transpose(2, 1, 0)
        A = np.random.randn(batch_size, 91, 3).astype(np.float32)
        expected_shape = (batch_size, 3, 2)

    A[..., 0] = 1
    Y = A @ T

    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    assert estimated_T.shape == expected_shape
    np.testing.assert_array_almost_equal(estimated_T, T, 5)
    assert estimated_T.dtype == np.float32


def test_aniostropic_similarity_transform() -> None:
    theta = 1.234
    tx, ty = -3, 5
    scale1, scale2 = 0.123, 0.456
    T = np.array(
        [
            [tx, np.cos(theta) * scale1, -np.sin(theta) * scale1],
            [ty, np.sin(theta) * scale2, np.cos(theta) * scale2],
        ],
        dtype=np.float32,
    ).T

    A = np.random.randn(91, 3).astype(np.float32)
    A[..., 0] = 1
    Y = A @ T

    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    np.testing.assert_array_almost_equal(estimated_T, T, 6)
