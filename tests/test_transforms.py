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

    rigid = register.transforms.RigidTransform()

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

    estimated_T = rigid.estimate(A, Y)
    assert estimated_T.shape == expected_shape
    np.testing.assert_array_almost_equal(estimated_T, T, 5)
    assert estimated_T.dtype == np.float32

    if batch_size is None:
        T_scaled = T.copy()
        T_scaled[1:] *= 2
        Y_scaled = A @ T_scaled
        estimated_T_scaled = rigid.estimate(A, Y_scaled)
        assert not np.allclose(estimated_T_scaled, T_scaled, atol=1e-5)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_similarity_transform(batch_size: int | None) -> None:
    T: NDArray[np.float32]
    A: NDArray[np.float32]
    expected_shape: tuple[int, ...]

    similarity = register.transforms.SimilarityTransform(reg=0.0)

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
        scales = np.random.uniform(0.1, 10, (batch_size,))
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

    estimated_T = similarity.estimate(A, Y)
    assert estimated_T.shape == expected_shape
    np.testing.assert_array_almost_equal(estimated_T, T, 5)
    assert estimated_T.dtype == np.float32


def test_similarity_transform_uses_least_squares_uniform_scale_with_noise() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2)).astype(np.float32)
    A = np.pad(X, [(0, 0), (1, 0)], constant_values=1).astype(np.float32)

    theta = 0.7
    scale = 2.0
    c, s = math.cos(theta), math.sin(theta)
    T = np.array([[1.0, -2.0], [c * scale, s * scale], [-s * scale, c * scale]], dtype=np.float32)
    Y_without_noise = A @ T

    noise = (rng.normal(size=Y_without_noise.shape) * 0.6).astype(np.float32)
    noise -= noise.mean(axis=0, keepdims=True)
    centered_signal = Y_without_noise - Y_without_noise.mean(axis=0, keepdims=True)
    noise -= centered_signal * (np.sum(centered_signal * noise) / np.sum(centered_signal * centered_signal))
    Y = Y_without_noise + noise

    similarity = register.transforms.SimilarityTransform(reg=0.0)
    estimated_T = similarity.estimate(A, Y)

    estimated_scale = np.sqrt(np.sum(estimated_T[1:] ** 2) / 2)
    assert estimated_scale == pytest.approx(scale, abs=1e-3)


def test_similarity_transform_does_not_fit_anisotropic_scale() -> None:
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

    similarity = register.transforms.SimilarityTransform(reg=0.0)
    estimated_T = similarity.estimate(A, Y)
    estimated_M = estimated_T[1:]
    estimated_MTM = estimated_M.T @ estimated_M

    assert not np.allclose(estimated_T, T, atol=1e-5)
    assert estimated_MTM[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert estimated_MTM[0, 0] == pytest.approx(estimated_MTM[1, 1], abs=1e-5)


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

    similarity = register.transforms.AnisotropicSimilarityTransform(reg=0.0)
    estimated_T = similarity.estimate(A, Y)
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

    similarity = register.transforms.AnisotropicSimilarityTransform(reg=0.0)
    estimated_T = similarity.estimate(A, Y)
    np.testing.assert_array_almost_equal(estimated_T, T, 5)


def test_estimate_thin_plate_spline_transform() -> None:
    np.random.seed(42)
    num_points = 50
    X = np.random.uniform(0, 1, (num_points, 2)).astype(np.float32)
    Y = X + np.random.normal(0, 0.01, (num_points, 2)).astype(np.float32)

    tps = register.transforms.ThinPlateSplineTransform(reg=0.0)
    A = tps.expand(X, Y)
    estimated_T = tps.estimate(A, Y)
    Y_hat = tps.transform_points(A, estimated_T)

    np.testing.assert_array_almost_equal(Y_hat, Y, 4)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_thin_plate_spline_transform_batch(batch_size: int | None) -> None:
    num_points = 50
    tps = register.transforms.ThinPlateSplineTransform(reg=0.0)

    if batch_size is None:
        X = np.random.uniform(0, 1, (num_points, 2)).astype(np.float32)
        Y = X + np.random.normal(0, 0.01, (num_points, 2)).astype(np.float32)
        A = tps.expand(X, Y)
        estimated_T = tps.estimate(A, Y)
        Y_hat = tps.transform_points(A, estimated_T)
        np.testing.assert_array_almost_equal(Y_hat, Y, 4)
    else:
        X = np.random.uniform(0, 1, (num_points, 2)).astype(np.float32)
        Y = X[None].repeat(batch_size, axis=0) + np.random.normal(0, 0.01, (batch_size, num_points, 2)).astype(
            np.float32
        )
        A_init = tps.expand(X, Y[0])
        A_batched = A_init[None].repeat(batch_size, axis=0)
        estimated_T = tps.estimate(A_batched, Y)
        Y_hat = tps.transform_points(A_batched, estimated_T)
        np.testing.assert_array_almost_equal(Y_hat, Y, 4)


def test_bernstein_basis_partition_of_unity() -> None:
    np.random.seed(42)
    X = np.random.uniform(0, 1, (100, 2)).astype(np.float32)

    bernstein = register.transforms.BernsteinTransform(degrees=2)
    A1 = bernstein.expand(X, X)
    assert A1.shape == (100, 9)
    np.testing.assert_array_almost_equal(A1.sum(axis=-1), np.ones(100), 6)


def test_bernstein_basis_nonnegative() -> None:
    np.random.seed(42)
    X = np.random.uniform(0, 1, (100, 2)).astype(np.float32)

    bernstein = register.transforms.BernsteinTransform(degrees=2)
    A2 = bernstein.expand(X, X)
    assert np.all(A2 >= 0)


def test_estimate_quadratic_bernstein_transform() -> None:
    np.random.seed(42)
    X = np.random.uniform(0, 1, (100, 2)).astype(np.float32)

    T_true = np.random.randn(9, 2).astype(np.float32) * 0.1

    bernstein = register.transforms.BernsteinTransform(degrees=2, reg=0.0)
    A = bernstein.expand(X, X)
    Y = (A @ T_true).astype(np.float32)

    estimated_T = bernstein.estimate(A, Y)
    assert estimated_T.shape == (9, 2)
    np.testing.assert_array_almost_equal(estimated_T, T_true, 5)
