import math

import numpy as np
import pytest
from numpy.typing import NDArray

import register.transforms


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_rigid_transform(batch_size: int | None) -> None:
    T: NDArray[np.float32]
    A: NDArray[np.float32]
    expected_shape: tuple[int, ...]

    if batch_size is None:
        theta = 1.234
        tx_val: float = -3.0
        ty_val: float = 5.0
        T = np.array(
            [
                [tx_val, math.cos(theta), -math.sin(theta)],
                [ty_val, math.sin(theta), math.cos(theta)],
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
def test_estimate_similarity_transform(batch_size: int | None) -> None:
    T: NDArray[np.float32]
    A: NDArray[np.float32]
    expected_shape: tuple[int, ...]

    if batch_size is None:
        theta = 1.234
        tx_val: float = -3.0
        ty_val: float = 5.0
        scale = 0.123
        T = np.array(
            [
                [tx_val, math.cos(theta) * scale, -math.sin(theta) * scale],
                [ty_val, math.sin(theta) * scale, math.cos(theta) * scale],
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


def test_anisotropic_similarity_transform_rotate_then_scale() -> None:
    theta = 1.234
    tx: float = -3.0
    ty: float = 5.0
    sx, sy = 0.123, 0.456
    c, s = math.cos(theta), math.sin(theta)

    D = np.array([[sx, 0], [0, sy]], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    M = R.T @ D
    t = np.array([[tx, ty]], dtype=np.float32)
    T = np.concatenate([t, M], axis=0)

    A = np.random.randn(91, 3).astype(np.float32)
    A[..., 0] = 1
    Y = A @ T

    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    np.testing.assert_array_almost_equal(estimated_T, T, 5)


def test_anisotropic_similarity_transform_scale_then_rotate() -> None:
    theta = 1.234
    tx: float = -3.0
    ty: float = 5.0
    sx, sy = 0.123, 0.456
    c, s = math.cos(theta), math.sin(theta)

    D = np.array([[sx, 0], [0, sy]], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    M = D @ R.T
    t = np.array([[tx, ty]], dtype=np.float32)
    T = np.concatenate([t, M], axis=0)

    A = np.random.randn(91, 3).astype(np.float32)
    A[..., 0] = 1
    Y = A @ T

    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    np.testing.assert_array_almost_equal(estimated_T, T, 5)


@pytest.mark.parametrize("lamb", [0.0, 0.1])
def test_estimate_thin_plate_spline_transform(lamb: float) -> None:
    np.random.seed(42)
    num_points = 50
    X = np.random.uniform(0, 1, (num_points, 2)).astype(np.float32)
    Y = X + np.random.normal(0, 0.01, (num_points, 2)).astype(np.float32)
    A = register.transforms.ThinPlateSplineTransform.expand(X, Y)
    estimated_T = register.transforms.ThinPlateSplineTransform.estimate(A, Y, lamb)
    Y_hat = register.transforms.ThinPlateSplineTransform.transform(A, estimated_T)

    if lamb == 0.0:
        np.testing.assert_array_almost_equal(Y_hat, Y, 4)
    else:
        W = estimated_T[:num_points, :2]
        P = np.concatenate([np.ones((num_points, 1)), X], axis=1)
        constraint = P.T @ W
        np.testing.assert_array_almost_equal(constraint, np.zeros((3, 2)), 6)
        np.testing.assert_allclose(Y_hat, Y, atol=0.02)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_thin_plate_spline_transform_batch(batch_size: int | None) -> None:
    num_points = 50
    if batch_size is None:
        X = np.random.uniform(0, 1, (num_points, 2)).astype(np.float32)
        Y = X + np.random.normal(0, 0.01, (num_points, 2)).astype(np.float32)
        A = register.transforms.ThinPlateSplineTransform.expand(X, Y)
        estimated_T = register.transforms.ThinPlateSplineTransform.estimate(A, Y, 0.0)
        Y_hat = register.transforms.ThinPlateSplineTransform.transform(A, estimated_T)
        np.testing.assert_array_almost_equal(Y_hat, Y, 4)
    else:
        X = np.random.uniform(0, 1, (num_points, 2)).astype(np.float32)
        Y = X[None].repeat(batch_size, axis=0) + np.random.normal(0, 0.01, (batch_size, num_points, 2)).astype(
            np.float32
        )
        A_init = register.transforms.ThinPlateSplineTransform.expand(X, Y[0])
        # In RANSAC, A is often just the expanded version of ALL points
        # But estimate/transform should handle whatever is passed.
        # Let's test the batch estimate
        # Note: estimate expects A and Y to Have same batch dimension if A is 3D
        A_batched = A_init[None].repeat(batch_size, axis=0)
        estimated_T = register.transforms.ThinPlateSplineTransform.estimate(A_batched, Y, 0.0)
        Y_hat = register.transforms.ThinPlateSplineTransform.transform(A_batched, estimated_T)
        np.testing.assert_array_almost_equal(Y_hat, Y, 4)
