import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import register.transforms


def uniform_weights(Y: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.ones(Y.shape[:-1], dtype=np.float32)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_rigid_transform(batch_size: int | None) -> None:
    rng = np.random.default_rng(0)
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
        A = rng.standard_normal((91, 3)).astype(np.float32)
        expected_shape = (3, 2)
    else:
        tx = rng.uniform(-10, 10, (batch_size,))
        ty = rng.uniform(-10, 10, (batch_size,))
        thetas = rng.uniform(-np.pi, np.pi, (batch_size,))
        T = np.array(
            [
                [tx, np.cos(thetas), -np.sin(thetas)],
                [ty, np.sin(thetas), np.cos(thetas)],
            ],
            dtype=np.float32,
        ).transpose(2, 1, 0)
        A = rng.standard_normal((batch_size, 91, 3)).astype(np.float32)
        expected_shape = (batch_size, 3, 2)

    A[..., 0] = 1
    Y = A @ T

    estimated_T = rigid.estimate(A, Y, uniform_weights(Y))
    assert estimated_T.shape == expected_shape
    np.testing.assert_array_almost_equal(estimated_T, T, 5)
    assert estimated_T.dtype == np.float32

    if batch_size is None:
        T_scaled = T.copy()
        T_scaled[1:] *= 2
        Y_scaled = A @ T_scaled
        estimated_T_scaled = rigid.estimate(A, Y_scaled, uniform_weights(Y_scaled))
        assert not np.allclose(estimated_T_scaled, T_scaled, atol=1e-5)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_similarity_transform(batch_size: int | None) -> None:
    rng = np.random.default_rng(0)
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
        A = rng.standard_normal((91, 3)).astype(np.float32)
        expected_shape = (3, 2)
    else:
        tx = rng.uniform(-10, 10, (batch_size,))
        ty = rng.uniform(-10, 10, (batch_size,))
        thetas = rng.uniform(-np.pi, np.pi, (batch_size,))
        scales = rng.uniform(0.1, 10, (batch_size,))
        T = np.array(
            [
                [tx, np.cos(thetas) * scales, -np.sin(thetas) * scales],
                [ty, np.sin(thetas) * scales, np.cos(thetas) * scales],
            ],
            dtype=np.float32,
        ).transpose(2, 1, 0)
        A = rng.standard_normal((batch_size, 91, 3)).astype(np.float32)
        expected_shape = (batch_size, 3, 2)

    A[..., 0] = 1
    Y = A @ T

    estimated_T = similarity.estimate(A, Y, uniform_weights(Y))
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
    estimated_T = similarity.estimate(A, Y, uniform_weights(Y))

    estimated_scale = np.sqrt(np.sum(estimated_T[1:] ** 2) / 2)
    assert estimated_scale == pytest.approx(scale, abs=1e-3)


def test_similarity_transform_does_not_fit_anisotropic_scale() -> None:
    rng = np.random.default_rng(0)
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

    A = rng.standard_normal((91, 3)).astype(np.float32)
    A[..., 0] = 1
    Y = A @ T

    similarity = register.transforms.SimilarityTransform(reg=0.0)
    estimated_T = similarity.estimate(A, Y, uniform_weights(Y))
    estimated_M = estimated_T[1:]
    estimated_MTM = estimated_M.T @ estimated_M

    assert not np.allclose(estimated_T, T, atol=1e-5)
    assert estimated_MTM[0, 1] == pytest.approx(0.0, abs=1e-5)
    assert estimated_MTM[0, 0] == pytest.approx(estimated_MTM[1, 1], abs=1e-5)


def test_anisotropic_similarity_transform_rotate_then_scale() -> None:
    rng = np.random.default_rng(0)
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

    A = rng.standard_normal((91, 3)).astype(np.float32)
    A[..., 0] = 1
    Y = A @ T

    similarity = register.transforms.AnisotropicSimilarityTransform(reg=0.0)
    estimated_T = similarity.estimate(A, Y, uniform_weights(Y))
    np.testing.assert_array_almost_equal(estimated_T, T, 5)


def test_anisotropic_similarity_transform_scale_then_rotate() -> None:
    rng = np.random.default_rng(0)
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

    A = rng.standard_normal((91, 3)).astype(np.float32)
    A[..., 0] = 1
    Y = A @ T

    similarity = register.transforms.AnisotropicSimilarityTransform(reg=0.0)
    estimated_T = similarity.estimate(A, Y, uniform_weights(Y))
    np.testing.assert_array_almost_equal(estimated_T, T, 5)


def test_bernstein_basis_partition_of_unity() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)

    bernstein = register.transforms.BernsteinTransform(degrees=2)
    A1 = bernstein.expand(X, X)
    assert A1.shape == (100, 9)
    np.testing.assert_array_almost_equal(A1.sum(axis=-1), np.ones(100), 6)


def test_bernstein_basis_nonnegative() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)

    bernstein = register.transforms.BernsteinTransform(degrees=2)
    A2 = bernstein.expand(X, X)
    assert np.all(A2 >= 0)


def test_estimate_quadratic_bernstein_transform() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)

    T_true = rng.standard_normal((9, 2)).astype(np.float32) * 0.1

    bernstein = register.transforms.BernsteinTransform(degrees=2, reg=0.0)
    A = bernstein.expand(X, X)
    Y = (A @ T_true).astype(np.float32)

    estimated_T = bernstein.estimate(A, Y, uniform_weights(Y))
    assert estimated_T.shape == (9, 2)
    np.testing.assert_array_almost_equal(estimated_T, T_true, 5)


@pytest.mark.parametrize(
    "transform",
    [
        register.transforms.PerspectiveTransform(),
        register.transforms.AffineTransform(),
        register.transforms.RigidTransform(),
        register.transforms.SimilarityTransform(),
        register.transforms.AnisotropicSimilarityTransform(),
        register.transforms.BernsteinTransform(degrees=2),
    ],
)
def test_estimate_weight_acts_like_point_multiplicity(transform: register.transforms.Transform) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 2)).astype(np.float32)
    Y = (X @ np.array([[0.9, 0.2], [-0.1, 1.1]]) + 0.1 + rng.normal(0, 0.05, X.shape)).astype(np.float32)
    A = transform.expand(X, Y)

    weights = np.ones(30, dtype=np.float32)
    weights[:5] = 2
    weights[-5:] = 0
    multiplicity = weights.astype(np.int64)

    weighted = transform.estimate(A, Y, weights)
    duplicated_Y = Y.repeat(multiplicity, axis=0)
    duplicated = transform.estimate(A.repeat(multiplicity, axis=0), duplicated_Y, uniform_weights(duplicated_Y))
    np.testing.assert_allclose(weighted, duplicated, atol=1e-4)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_perspective_transform(batch_size: int | None) -> None:
    rng = np.random.default_rng(0)
    batch_shape = () if batch_size is None else (batch_size,)
    H = np.eye(3) + rng.normal(0, 0.1, (*batch_shape, 3, 3))
    H /= H[..., -1:, -1:]
    X = rng.uniform(0, 1, (*batch_shape, 20, 2)).astype(np.float32)
    Y_homogeneous = np.pad(X, [(0, 0)] * (X.ndim - 1) + [(0, 1)], constant_values=1) @ H.swapaxes(-2, -1)
    Y = (Y_homogeneous[..., :2] / Y_homogeneous[..., 2:]).astype(np.float32)

    perspective = register.transforms.PerspectiveTransform()
    A = perspective.expand(X, Y)
    estimated_T = perspective.estimate(A, Y, uniform_weights(Y))

    assert estimated_T.shape == (*batch_shape, 3, 3)
    np.testing.assert_allclose(estimated_T, H.swapaxes(-2, -1), atol=1e-3)
    np.testing.assert_allclose(perspective.transform_points(A, estimated_T), Y, atol=1e-4)


def similarity_then_shear(shear: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Row-vector linear part (x @ L) of a 1-radian rotation scaled by 0.7, followed by `shear`, plus a translation."""
    c, s = math.cos(1.0), math.sin(1.0)
    L = 0.7 * np.array([[c, s], [-s, c]]) @ np.array([[1, shear], [0, 1]])
    return L, np.array([0.3, -0.2])


def bernstein_params(degrees: int, L: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float32]:
    steps = np.linspace(0, 1, degrees + 1)
    grid = np.stack(np.meshgrid(steps, steps), axis=-1).reshape(-1, 2)
    return (grid @ L + t).astype(np.float32)


def perspective_params(L: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float32]:
    return np.block([[L, np.zeros((2, 1))], [t, 1]]).astype(np.float32)


@pytest.mark.parametrize(
    ("transform", "to_params"),
    [
        (register.transforms.AffineTransform(), lambda L, t: np.vstack([t, L]).astype(np.float32)),
        (register.transforms.BernsteinTransform(degrees=2), lambda L, t: bernstein_params(2, L, t)),
        (register.transforms.BernsteinTransform(degrees=3), lambda L, t: bernstein_params(3, L, t)),
        (register.transforms.PerspectiveTransform(), perspective_params),
    ],
)
def test_penalty_ignores_similarity_and_charges_shear(
    transform: register.transforms.Transform,
    to_params: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float32]],
) -> None:
    assert transform.penalty(to_params(*similarity_then_shear(0.0))) == pytest.approx(0.0, abs=1e-6)
    assert transform.penalty(to_params(*similarity_then_shear(0.3))) > 1e-2


@pytest.mark.parametrize(
    "transform",
    [register.transforms.AffineTransform(reg=1.0), register.transforms.BernsteinTransform(degrees=2, reg=1.0)],
)
def test_strong_regularization_does_not_bias_similarity_fits(transform: register.transforms.Transform) -> None:
    rng = np.random.default_rng(0)
    L, t = similarity_then_shear(0.0)
    X = rng.uniform(0, 1, (40, 2)).astype(np.float32)
    Y = (X @ L + t).astype(np.float32)

    A = transform.expand(X, Y)
    estimated_T = transform.estimate(A, Y, uniform_weights(Y))
    np.testing.assert_allclose(transform.transform_points(A, estimated_T), Y, atol=1e-4)


@pytest.mark.parametrize(
    ("transform", "expected_singular_values"),
    [
        (register.transforms.RigidTransform(), lambda s: np.array([1.0, 1.0])),
        (register.transforms.SimilarityTransform(), lambda s: np.full(2, s.mean())),
    ],
)
def test_fit_respects_model_constraints(
    transform: register.transforms.Transform,
    expected_singular_values: Callable[[NDArray[np.float32]], NDArray[np.float64]],
) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (60, 2)).astype(np.float32)
    Y = (X @ similarity_then_shear(0.4)[0] + 0.1).astype(np.float32)

    transform.fit(X, Y, np.ones(len(X), dtype=np.float32))
    singular_values = np.linalg.svd(transform.matrix_[1:], compute_uv=False)
    np.testing.assert_allclose(singular_values, expected_singular_values(singular_values), atol=1e-4)


def mirrored(X: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.stack([1 - X[:, 0], X[:, 1]], axis=-1)


def folded(X: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.stack([(X[:, 0] - 0.5) ** 2, X[:, 1]], axis=-1)


def beyond_horizon(X: NDArray[np.float32]) -> NDArray[np.float32]:
    # Homography whose horizon (zero projective depth) is the line x = 0.5.
    return X / (1 - 2 * X[:, :1])


@pytest.mark.parametrize(
    ("transform", "warp", "valid"),
    [
        (register.transforms.AffineTransform(), lambda X: X @ similarity_then_shear(0.2)[0], True),
        (register.transforms.PerspectiveTransform(), lambda X: X @ similarity_then_shear(0.2)[0], True),
        (register.transforms.BernsteinTransform(degrees=2), lambda X: X @ similarity_then_shear(0.2)[0], True),
        (register.transforms.AffineTransform(), mirrored, False),
        (register.transforms.PerspectiveTransform(), mirrored, False),
        (register.transforms.BernsteinTransform(degrees=2), folded, False),
        (register.transforms.PerspectiveTransform(), beyond_horizon, False),
    ],
)
def test_fit_reports_infinite_loss_for_invalid_warps(
    transform: register.transforms.Transform,
    warp: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    valid: bool,
) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (60, 2)).astype(np.float32)
    Y = warp(X).astype(np.float32)

    transform.fit(X, Y, np.ones(len(X), dtype=np.float32))
    assert math.isfinite(transform.loss_) == valid


def collinear(n: int) -> NDArray[np.float32]:
    # Points on y = 0: every homography sample drawn from them has H[2, 2] = 0, so normalizing it gives NaNs.
    return np.stack([np.arange(n), np.zeros(n)], axis=-1).astype(np.float32)


def test_fit_skips_degenerate_hypotheses() -> None:
    rng = np.random.default_rng(0)
    X = np.concatenate([collinear(4), rng.uniform(0, 1, (16, 2)).astype(np.float32)])
    weights = np.linspace(1, 0.5, len(X), dtype=np.float32)  # the collinear points form the first RANSAC sample

    transform = register.transforms.PerspectiveTransform().fit(X, X, weights)
    np.testing.assert_allclose(transform.matrix_, np.eye(3), atol=1e-4)


def test_fit_reports_infinite_loss_when_every_hypothesis_is_degenerate() -> None:
    X = collinear(4)  # the minimal sample size, so every hypothesis is fitted to the same degenerate points
    transform = register.transforms.PerspectiveTransform().fit(X, X, np.ones(len(X), dtype=np.float32))
    assert transform.loss_ == math.inf


def test_irls_rejects_non_finite_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (20, 2)).astype(np.float32)
    transform = register.transforms.AffineTransform()
    # RANSAC fits samples with estimate_sample; only IRLS steps go through estimate.
    monkeypatch.setattr(transform, "estimate", lambda A, Y, weights: np.full((3, 2), np.nan, dtype=np.float32))

    transform.fit(X, X, np.ones(len(X), dtype=np.float32))
    assert math.isfinite(transform.loss_)
    np.testing.assert_allclose(transform.matrix_, [[0, 0], [1, 0], [0, 1]], atol=1e-4)
