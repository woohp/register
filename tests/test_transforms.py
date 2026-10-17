import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from register import transforms


def uniform_weights(Y: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.ones(Y.shape[:-1], dtype=np.float32)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_rigid_transform(batch_size: int | None) -> None:
    rng = np.random.default_rng(0)
    T: NDArray[np.float32]
    A: NDArray[np.float32]
    expected_shape: tuple[int, ...]

    rigid = transforms.RigidTransform()

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

    similarity = transforms.SimilarityTransform(reg=0.0)

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
    T = np.array(
        [[1.0, -2.0], [c * scale, s * scale], [-s * scale, c * scale]], dtype=np.float32
    )
    Y_without_noise = A @ T

    noise = (rng.normal(size=Y_without_noise.shape) * 0.6).astype(np.float32)
    noise -= noise.mean(axis=0, keepdims=True)
    centered_signal = Y_without_noise - Y_without_noise.mean(axis=0, keepdims=True)
    noise -= centered_signal * (
        np.sum(centered_signal * noise) / np.sum(centered_signal * centered_signal)
    )
    Y = Y_without_noise + noise

    similarity = transforms.SimilarityTransform(reg=0.0)
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

    similarity = transforms.SimilarityTransform(reg=0.0)
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

    anisotropic = transforms.AnisotropicSimilarityTransform(reg=0.0)
    estimated_T = anisotropic.estimate(A, Y, uniform_weights(Y))
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

    anisotropic = transforms.AnisotropicSimilarityTransform(reg=0.0)
    estimated_T = anisotropic.estimate(A, Y, uniform_weights(Y))
    np.testing.assert_array_almost_equal(estimated_T, T, 5)


@pytest.mark.parametrize("scale_first", [True, False])
def test_anisotropic_similarity_transform_finds_the_least_squares_fit(
    scale_first: bool,
) -> None:
    # A thin, correlated cloud with noise, where projecting the affine fit onto rotation
    # and scales falls short.
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((50, 2)) @ [[1, 0.8], [0, 0.3]]).astype(np.float32)
    c, s = math.cos(0.7), math.sin(0.7)
    R, D = np.array([[c, s], [-s, c]]), np.diag([1.3, 0.6])
    Y = (
        X @ (D @ R if scale_first else R @ D) + 0.1 * rng.standard_normal(X.shape)
    ).astype(np.float32)

    transform = transforms.AnisotropicSimilarityTransform()
    A = transform.expand(X, Y)
    T = transform.estimate(A, Y, uniform_weights(Y))
    M = T[1:]
    # Orthogonal rows (D R) or columns (R D).
    assert min(abs(M[0] @ M[1]), abs(M[:, 0] @ M[:, 1])) < 1e-5
    error = ((A.astype(np.float64) @ T - Y) ** 2).sum()
    assert error <= brute_force_anisotropic_error(X, Y) * (1 + 1e-5)


def brute_force_anisotropic_error(
    X: NDArray[np.float32], Y: NDArray[np.float32]
) -> float:
    """Least squared error over rotation angles, for both D R and R D.

    Each rotation gets its best scales.
    """
    X_centered = X - X.mean(axis=0, dtype=np.float64)
    Y_centered = Y - Y.mean(axis=0, dtype=np.float64)
    S, B = X_centered.T @ X_centered, X_centered.T @ Y_centered
    angles = np.linspace(-np.pi, np.pi, 20_000, endpoint=False)
    c, s = np.cos(angles), np.sin(angles)
    R = np.stack([np.stack([c, s], axis=-1), np.stack([-s, c], axis=-1)], axis=-2)
    # Each scale is a 1-D least-squares fit,
    # removing (covariance with its target)^2 / (variance) from the error.
    scale_first = (R * B).sum(axis=-1) ** 2 / np.diagonal(S)
    rotate_first = (R * B).sum(axis=-2) ** 2 / np.einsum("gij,ik,gkj->gj", R, S, R)
    explained = np.maximum(scale_first.sum(axis=-1), rotate_first.sum(axis=-1)).max()
    return float((Y_centered**2).sum() - explained)


@pytest.mark.parametrize(
    ("S", "B", "optimum"),
    [
        # B has no component along S's weakest axis that the multiplier could balance,
        # so the optimum sits at the end of the multiplier's interval;
        # stopping the bisection there would leave m1 . m2 = 1/9.
        (np.diag([1.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0]]), -2 / 3),
        # Nearly that, with nearly equal eigenvalues: the root is within rounding of the
        # interval's end, where a rounded multiplier would leave m1 . m2 = -1e-4.
        (np.diag([1.0, 1 + 1e-15]), np.array([[0.5, 0.5], [5e-16, -5e-16]]), -0.25),
    ],
)
def test_anisotropic_rotate_then_scale_solves_the_hard_case(
    S: NDArray[np.float64], B: NDArray[np.float64], optimum: float
) -> None:
    M = transforms._rotate_then_scale(S, B)
    assert M[:, 0] @ M[:, 1] == pytest.approx(0, abs=1e-12)
    # The brute-force optimum.
    assert np.trace(M.T @ S @ M) - 2 * np.trace(M.T @ B) == pytest.approx(optimum)


def test_anisotropic_similarity_transform_prefers_the_better_mirrored_fit() -> None:
    # Both orders' optima are mirrored here, so neither is preferred for orientation;
    # the better fit must still win (fit then reports an infinite loss).
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((50, 2)) @ [[1, 0.8], [0, 0.3]]).astype(np.float32)
    c, s = math.cos(0.7), math.sin(0.7)
    M = np.array([[c, s], [-s, c]]) @ np.diag([1.3, -0.6])
    Y = (X @ M).astype(np.float32)

    transform = transforms.AnisotropicSimilarityTransform()
    T = transform.estimate(transform.expand(X, Y), Y, uniform_weights(Y))
    assert np.allclose(T[1:], M, atol=1e-5)


def test_bernstein_basis_partition_of_unity() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)

    bernstein = transforms.BernsteinTransform(degrees=2)
    A = bernstein.expand(X, X)
    assert A.shape == (100, 9)
    np.testing.assert_array_almost_equal(A.sum(axis=-1), np.ones(100), 6)


def test_bernstein_basis_nonnegative() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)

    bernstein = transforms.BernsteinTransform(degrees=2)
    A = bernstein.expand(X, X)
    assert np.all(A >= 0)


def test_estimate_quadratic_bernstein_transform() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (100, 2)).astype(np.float32)

    T_true = rng.standard_normal((9, 2)).astype(np.float32) * 0.1

    bernstein = transforms.BernsteinTransform(degrees=2, reg=0.0)
    A = bernstein.expand(X, X)
    Y = (A @ T_true).astype(np.float32)

    estimated_T = bernstein.estimate(A, Y, uniform_weights(Y))
    assert estimated_T.shape == (9, 2)
    np.testing.assert_array_almost_equal(estimated_T, T_true, 5)


@pytest.mark.parametrize(
    "transform",
    [
        transforms.PerspectiveTransform(),
        transforms.AffineTransform(),
        transforms.RigidTransform(),
        transforms.SimilarityTransform(),
        transforms.AnisotropicSimilarityTransform(),
        transforms.BernsteinTransform(degrees=2),
    ],
)
def test_estimate_weight_acts_like_point_multiplicity(
    transform: transforms.Transform,
) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 2)).astype(np.float32)
    Y = (
        X @ np.array([[0.9, 0.2], [-0.1, 1.1]]) + 0.1 + rng.normal(0, 0.05, X.shape)
    ).astype(np.float32)
    A = transform.expand(X, Y)

    weights = np.ones(30, dtype=np.float32)
    weights[:5] = 2
    weights[-5:] = 0
    multiplicity = weights.astype(np.int64)

    weighted = transform.estimate(A, Y, weights)
    duplicated_Y = Y.repeat(multiplicity, axis=0)
    duplicated = transform.estimate(
        A.repeat(multiplicity, axis=0), duplicated_Y, uniform_weights(duplicated_Y)
    )
    np.testing.assert_allclose(weighted, duplicated, atol=1e-4)


@pytest.mark.parametrize("batch_size", [None, 5])
def test_estimate_perspective_transform(batch_size: int | None) -> None:
    rng = np.random.default_rng(0)
    batch_shape = () if batch_size is None else (batch_size,)
    H = np.eye(3) + rng.normal(0, 0.1, (*batch_shape, 3, 3))
    H /= H[..., -1:, -1:]
    X = rng.uniform(0, 1, (*batch_shape, 20, 2)).astype(np.float32)
    X_homogeneous = np.pad(X, [(0, 0)] * (X.ndim - 1) + [(0, 1)], constant_values=1)
    Y_homogeneous = X_homogeneous @ H.swapaxes(-2, -1)
    Y = (Y_homogeneous[..., :2] / Y_homogeneous[..., 2:]).astype(np.float32)

    perspective = transforms.PerspectiveTransform()
    A = perspective.expand(X, Y)
    estimated_T = perspective.estimate(A, Y, uniform_weights(Y))

    assert estimated_T.shape == (*batch_shape, 3, 3)
    np.testing.assert_allclose(estimated_T, H.swapaxes(-2, -1), atol=1e-3)
    np.testing.assert_allclose(
        perspective.transform_points(A, estimated_T), Y, atol=1e-4
    )


@pytest.mark.parametrize(
    "transform",
    [
        transforms.AffineTransform(),
        transforms.BernsteinTransform(degrees=2),
        transforms.PerspectiveTransform(),
    ],
)
def test_squared_residuals_match_transformed_points(
    transform: transforms.Transform,
) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 2)).astype(np.float32)
    # Inlier-scale noise: squared residuals are ~1e-5 of the squared coordinates,
    # small enough that computing them by cancellation in float32 would fail.
    Y = (X + rng.normal(0, 0.003, X.shape)).astype(np.float32)
    A = transform.expand(X, Y)
    samples = rng.choice(len(X), (5, transform.min_samples))
    Ts = transform.estimate_sample(A[samples], Y[samples])

    for T in [Ts, Ts[0]]:
        expected = ((transform.transform_points(A, T) - Y) ** 2).sum(-1)
        np.testing.assert_allclose(
            transform.squared_residuals(A, T, Y), expected, rtol=1e-3, atol=1e-10
        )


def similarity_then_shear(
    shear: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Row-vector linear part (x @ L) and translation of a sheared similarity.

    The similarity is a 1-radian rotation scaled by 0.7, followed by `shear`.
    """
    c, s = math.cos(1.0), math.sin(1.0)
    L = 0.7 * np.array([[c, s], [-s, c]]) @ np.array([[1, shear], [0, 1]])
    return L, np.array([0.3, -0.2])


def bernstein_params(
    degrees: int, L: NDArray[np.float64], t: NDArray[np.float64]
) -> NDArray[np.float32]:
    steps = np.linspace(0, 1, degrees + 1)
    grid = np.stack(np.meshgrid(steps, steps), axis=-1).reshape(-1, 2)
    return (grid @ L + t).astype(np.float32)


def perspective_params(
    L: NDArray[np.float64], t: NDArray[np.float64]
) -> NDArray[np.float32]:
    return np.block([[L, np.zeros((2, 1))], [t, 1]]).astype(np.float32)


@pytest.mark.parametrize(
    ("transform", "to_params"),
    [
        (
            transforms.AffineTransform(),
            lambda L, t: np.vstack([t, L]).astype(np.float32),
        ),
        (
            transforms.BernsteinTransform(degrees=2),
            lambda L, t: bernstein_params(2, L, t),
        ),
        (
            transforms.BernsteinTransform(degrees=3),
            lambda L, t: bernstein_params(3, L, t),
        ),
        (transforms.PerspectiveTransform(), perspective_params),
    ],
)
def test_penalty_ignores_similarity_and_charges_shear(
    transform: transforms.Transform,
    to_params: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float32]
    ],
) -> None:
    similarity = to_params(*similarity_then_shear(0.0))
    sheared = to_params(*similarity_then_shear(0.3))
    assert transform.penalty(similarity) == pytest.approx(0.0, abs=1e-6)
    assert transform.penalty(sheared) > 1e-2


@pytest.mark.parametrize(
    "transform",
    [
        transforms.AffineTransform(reg=1.0),
        transforms.BernsteinTransform(degrees=2, reg=1.0),
    ],
)
def test_strong_regularization_does_not_bias_similarity_fits(
    transform: transforms.Transform,
) -> None:
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
        (transforms.RigidTransform(), lambda s: np.array([1.0, 1.0])),
        (transforms.SimilarityTransform(), lambda s: np.full(2, s.mean())),
    ],
)
def test_fit_respects_model_constraints(
    transform: transforms.Transform,
    expected_singular_values: Callable[[NDArray[np.float32]], NDArray[np.float64]],
) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (60, 2)).astype(np.float32)
    Y = (X @ similarity_then_shear(0.4)[0] + 0.1).astype(np.float32)

    transform.fit(X, Y, np.ones(len(X), dtype=np.float32))
    singular_values = np.linalg.svd(transform.matrix_[1:], compute_uv=False)
    np.testing.assert_allclose(
        singular_values, expected_singular_values(singular_values), atol=1e-4
    )


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
        (
            transforms.AffineTransform(),
            lambda X: X @ similarity_then_shear(0.2)[0],
            True,
        ),
        (
            transforms.PerspectiveTransform(),
            lambda X: X @ similarity_then_shear(0.2)[0],
            True,
        ),
        (
            transforms.BernsteinTransform(degrees=2),
            lambda X: X @ similarity_then_shear(0.2)[0],
            True,
        ),
        (transforms.AffineTransform(), mirrored, False),
        (transforms.PerspectiveTransform(), mirrored, False),
        (transforms.AnisotropicSimilarityTransform(), mirrored, False),
        (transforms.BernsteinTransform(degrees=2), folded, False),
        (transforms.PerspectiveTransform(), beyond_horizon, False),
    ],
)
def test_fit_reports_infinite_loss_for_invalid_warps(
    transform: transforms.Transform,
    warp: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    valid: bool,
) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (60, 2)).astype(np.float32)
    Y = warp(X).astype(np.float32)

    transform.fit(X, Y, np.ones(len(X), dtype=np.float32))
    assert math.isfinite(transform.loss_) == valid


def collinear(n: int) -> NDArray[np.float32]:
    # Points on y = 0 leave a homography sample underdetermined (a 4-D null space),
    # and the SVD returns a null vector with H[2, 2] = 0, so normalizing it gives NaNs.
    return np.stack([np.arange(n), np.zeros(n)], axis=-1).astype(np.float32)


def test_fit_skips_degenerate_hypotheses() -> None:
    rng = np.random.default_rng(0)
    X = np.concatenate([collinear(4), rng.uniform(0, 1, (16, 2)).astype(np.float32)])
    # The collinear points form the first RANSAC sample.
    weights = np.linspace(1, 0.5, len(X), dtype=np.float32)

    transform = transforms.PerspectiveTransform().fit(X, X, weights)
    np.testing.assert_allclose(transform.matrix_, np.eye(3), atol=1e-4)


def test_fit_reports_infinite_loss_when_every_hypothesis_is_degenerate() -> None:
    # The minimal sample size: every hypothesis is fitted to the same degenerate points.
    X = collinear(4)
    transform = transforms.PerspectiveTransform().fit(
        X, X, np.ones(len(X), dtype=np.float32)
    )
    assert transform.loss_ == math.inf


def test_irls_rejects_non_finite_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (20, 2)).astype(np.float32)
    transform = transforms.AffineTransform()
    # RANSAC fits samples with estimate_sample; only IRLS steps go through estimate.
    monkeypatch.setattr(
        transform,
        "estimate",
        lambda A, Y, weights: np.full((3, 2), np.nan, dtype=np.float32),
    )

    transform.fit(X, X, np.ones(len(X), dtype=np.float32))
    assert math.isfinite(transform.loss_)
    np.testing.assert_allclose(transform.matrix_, [[0, 0], [1, 0], [0, 1]], atol=1e-4)


def test_orientation_check_catches_a_horizon_at_the_bounding_box_edge() -> None:
    # x' = -1 / (x - 1e-4), y' = y / (x - 1e-4): the horizon x = 1e-4 lies just inside
    # the box [0, 1]^2, closer to its edge than a finite-difference step.
    # Check a known homography directly: fitting points near this horizon is
    # unstable in float32.
    H = np.array([[0, 0, -1], [0, 1, 0], [1, 0, -1e-4]], dtype=np.float32)
    X = np.array([[0, 0], [1, 1]], dtype=np.float32)
    perspective = transforms.PerspectiveTransform()
    assert not perspective._preserves_orientation(X, H.T)
    assert perspective._preserves_orientation(X + 0.5, H.T)
