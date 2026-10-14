from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import get_distance_squared
from .utils.sampling import get_random_samples

f32Array = NDArray[np.float32]
u8Array = NDArray[np.uint8]

# Tiny ridge to keep degenerate RANSAC samples (e.g. repeated points) solvable.
_CONDITIONING_RIDGE = 1e-6


@dataclass
class Transform(ABC):
    """Base class for geometric transforms with RANSAC + IRLS fitting.

    `reg` weights `penalty`, which measures distortion beyond a similarity (shear,
    stretch, bending). Rotation, uniform scale and translation are never penalized.

    Follows scikit-learn conventions: the constructor only stores settings,
    and `fit` returns self and sets the learned attributes `matrix_` (transform
    parameters) and `loss_`, which do not exist before fitting.
    """

    reg: float = 0.005
    sigma: float = 0.00003
    n_iters_ransac: int = 4_000
    n_iters_irls: int = 50
    matrix_: f32Array = field(init=False, repr=False, compare=False)
    loss_: float = field(init=False, repr=False, compare=False)

    @property
    @abstractmethod
    def min_samples(self) -> int:
        """Minimum number of points needed to estimate the transform."""
        ...

    @abstractmethod
    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        """Build design matrix A from source points X."""
        ...

    @abstractmethod
    def estimate(self, A: f32Array, Y: f32Array, weights: f32Array) -> f32Array:
        """Estimate transform parameters by weighted least squares.

        A is the design matrix and Y the target points. `weights` has shape Y.shape[:-1]
        and scales each point's squared residual.
        """
        ...

    def estimate_sample(self, A: f32Array, Y: f32Array) -> f32Array:
        """Estimate RANSAC hypotheses from (batched) minimal samples.

        Their penalty is applied when scoring them.
        """
        return self.estimate(A, Y, np.ones(Y.shape[:-1], dtype=np.float32))

    @staticmethod
    @abstractmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        """Apply transform T to points in design matrix A."""
        ...

    @abstractmethod
    def penalty(self, T: f32Array) -> f32Array:
        """Regularization penalty for transform parameters T."""
        ...

    @abstractmethod
    def warp(self, image: u8Array, template_shape: tuple[int, int]) -> u8Array:
        """Warp `image` into the frame of a template of shape `template_shape`."""
        ...

    def fit(self, X: f32Array, Y: f32Array, weights: f32Array) -> Self:
        """Fit the transform from points X to Y with per-point weights.

        Correspondences must be sorted by decreasing match quality: RANSAC draws its
        early hypotheses from a growing prefix of the points (PROSAC-style),
        so unsorted input silently degrades the fit.

        `loss_` is the robust loss plus penalty, for callers to threshold.
        It is inf when every RANSAC hypothesis is degenerate, or when the fitted warp is
        not a valid page mapping over the bounding box of X: it folds, mirrors,
        or (for perspective) places the horizon inside that box. The check is exact for
        affine-family and perspective transforms; for Bernstein it samples a grid,
        so folds smaller than its spacing can slip through. `matrix_` is set regardless.
        """
        A = self.expand(X, Y)
        T, loss = self._ransac(A, Y, weights)

        if math.isfinite(loss):  # otherwise every hypothesis was degenerate
            T, loss = self._irls(A, Y, weights, T, loss)
            if not self._preserves_orientation(X, T):
                loss = math.inf

        self.matrix_ = T
        self.loss_ = loss
        return self

    def _preserves_orientation(self, X: f32Array, T: f32Array) -> bool:
        """Whether the warp's Jacobian determinant is positive over X's bounding box.

        This generic version samples a grid with finite differences,
        so it can miss folds between grid points. Models with an exact test override it.
        """
        steps = np.linspace(0, 1, 16, dtype=np.float32)
        unit_grid = np.stack(np.meshgrid(steps, steps), axis=-1).reshape(-1, 2)
        grid = unit_grid * np.ptp(X, axis=0) + X.min(axis=0)
        h = np.float32(1e-3)
        points = np.concatenate([grid, grid + [h, 0], grid + [0, h]]).astype(np.float32)
        mapped = self.transform_points(self.expand(points, points), T)
        origin, dx, dy = np.split(mapped, 3)
        dx, dy = dx - origin, dy - origin
        return bool(np.all(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0] > 0))

    def _geman_mcclure_loss(self, d: f32Array) -> f32Array:
        return (d / (d + self.sigma)).astype(np.float32)

    def _ransac(
        self, A: f32Array, Y: f32Array, weights: f32Array
    ) -> tuple[f32Array, float]:
        samples = get_random_samples(
            Y.shape[0], self.min_samples, self.n_iters_ransac, random.Random(4321)
        )

        Ts = self.estimate_sample(A[samples], Y[samples])

        distance_squared = get_distance_squared(self.transform_points(A, Ts), Y)
        losses = self._loss(distance_squared, weights, Ts)
        # Degenerate samples (e.g. collinear points for a homography) can give
        # non-finite hypotheses, and argmin would pick a NaN.
        losses = np.where(np.isfinite(losses), losses, np.inf)

        best_idx = np.argmin(losses)
        return Ts[best_idx], losses[best_idx].item()

    def _irls(
        self, A: f32Array, Y: f32Array, weights: f32Array, T: f32Array, loss: float
    ) -> tuple[f32Array, float]:
        normalized_weights = weights / weights.sum()
        distance_squared = get_distance_squared(self.transform_points(A, T), Y)

        for _ in range(self.n_iters_irls):
            # Weighting each squared residual by the derivative of the Geman-McClure
            # loss gives a quadratic majorizer of the robust objective,
            # so the weighted least-squares fit is the next iterate.
            irls_weights = (
                normalized_weights * self.sigma / (distance_squared + self.sigma) ** 2
            )
            new_T = self.estimate(A, Y, irls_weights)
            new_distance_squared = get_distance_squared(
                self.transform_points(A, new_T), Y
            )
            new_loss = self._loss(new_distance_squared, weights, new_T).item()

            if not new_loss < loss:  # also rejects a non-finite step
                break

            converged = loss - new_loss < 1e-6
            T, loss, distance_squared = new_T, new_loss, new_distance_squared
            if converged:
                break

        return T, loss

    def _loss(
        self, distance_squared: f32Array, weights: f32Array, T: f32Array
    ) -> NDArray[np.floating]:
        objective = np.average(
            self._geman_mcclure_loss(distance_squared), axis=-1, weights=weights
        )
        return objective + self.reg * self.penalty(T)


@dataclass
class _LinearTransform(Transform):
    """Internal base for transforms that are linear in their parameters (Y = A @ T).

    The penalty is the squared distance of T from the subspace of similarity transforms.
    It is quadratic, so `estimate` minimizes weighted squared error + reg * penalty in
    closed form, matching `_loss`.
    """

    @abstractmethod
    def similarity_basis(self) -> f32Array:
        """Parameters of the x/y translations, the identity, and the 90° rotation.

        Shape (4, *T.shape).
        """
        ...

    def estimate(self, A: f32Array, Y: f32Array, weights: f32Array) -> f32Array:
        n_params = A.shape[-1]
        A_T_W = A.swapaxes(-2, -1) * weights[..., None, :]

        # Solve for vec(T) (row-major): squared error treats output columns separately,
        # the penalty couples them.
        gram = (
            (A_T_W @ A)[..., :, None, :, None] * np.eye(2, dtype=A.dtype)[:, None, :]
        ).reshape(*A.shape[:-2], 2 * n_params, 2 * n_params)
        lhs = (
            gram
            + self.reg * self._non_similarity_projector()
            + _CONDITIONING_RIDGE * np.eye(2 * n_params)
        )

        rhs = (A_T_W @ Y).reshape(*A.shape[:-2], 2 * n_params, 1)
        vec_T = np.linalg.solve(lhs, rhs)
        return vec_T.reshape(*A.shape[:-2], n_params, 2).astype(np.float32)

    def estimate_sample(self, A: f32Array, Y: f32Array) -> f32Array:
        # Fit without the similarity penalty so extreme hypotheses remain extreme
        # and can be rejected by scoring. The conditioning ridge keeps degenerate
        # samples solvable; output columns can be fitted independently.
        A_T = A.swapaxes(-2, -1)
        lhs = A_T @ A + _CONDITIONING_RIDGE * np.eye(A.shape[-1], dtype=A.dtype)
        return cast(f32Array, np.linalg.solve(lhs, A_T @ Y))

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        return A @ T

    def penalty(self, T: f32Array) -> f32Array:
        vec_T = T.reshape(*T.shape[:-2], -1)
        return np.einsum(
            "...i,ij,...j->...", vec_T, self._non_similarity_projector(), vec_T
        )

    def _non_similarity_projector(self) -> f32Array:
        basis = self.similarity_basis().reshape(4, -1).T.astype(np.float64)
        projector = np.eye(basis.shape[0]) - basis @ np.linalg.pinv(basis)
        return projector.astype(np.float32)


@dataclass
class AffineTransform(_LinearTransform):
    """6-DOF affine transform: translation, rotation, scale, shear."""

    @property
    def min_samples(self) -> int:
        return 3

    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        return np.pad(X, [(0, 0)] * (X.ndim - 1) + [(1, 0)], constant_values=1)

    def similarity_basis(self) -> f32Array:
        return _similarity_basis(
            identity=np.array([[0, 0], [1, 0], [0, 1]]), translation=np.array([1, 0, 0])
        )

    def _preserves_orientation(self, X: f32Array, T: f32Array) -> bool:
        return bool(np.linalg.det(T[1:]) > 0)

    def warp(self, image: u8Array, template_shape: tuple[int, int]) -> u8Array:
        T = self.matrix_.copy()
        image_max_size = max(image.shape[:2])
        template_max_size = max(template_shape)
        T[0] *= image_max_size
        T[1:] *= image_max_size / template_max_size
        M = np.roll(T, -1, axis=0).T
        return cast(
            u8Array,
            cv2.warpAffine(
                image,
                M,
                dsize=template_shape[::-1],
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            ),
        )


def _similarity_basis(
    identity: NDArray[np.floating], translation: NDArray[np.floating]
) -> f32Array:
    """Stack the x/y translations, `identity`, and `identity` rotated by 90°.

    Returns a (4, K, 2) basis.
    """
    zeros = np.zeros_like(translation)
    x_shift = np.stack([translation, zeros], axis=-1)
    y_shift = np.stack([zeros, translation], axis=-1)
    rotated = identity @ np.array([[0, 1], [-1, 0]])
    return np.stack([x_shift, y_shift, identity, rotated]).astype(np.float32)


@dataclass
class _CenteredAffineTransform(AffineTransform):
    """Internal base for affine-like transforms estimated around centroids.

    These models already constrain deformation: rigid and similarity cannot shear,
    and independent stretch is intentional in the anisotropic model. No distortion
    penalty is added.
    """

    def estimate(self, A: f32Array, Y: f32Array, weights: f32Array) -> f32Array:
        src_xy = A[..., 1:]
        w = weights[..., None]

        w_sum = w.sum(axis=-2, keepdims=True)
        src_center = (w * src_xy).sum(axis=-2, keepdims=True) / w_sum
        dst_center = (w * Y).sum(axis=-2, keepdims=True) / w_sum

        # Scaling centered points by sqrt(w) turns every unweighted moment below into
        # its weighted counterpart.
        sqrt_w = np.sqrt(w)
        M = self.estimate_centered(
            sqrt_w * (src_xy - src_center), sqrt_w * (Y - dst_center)
        )
        t = dst_center - src_center @ M

        return np.concatenate([t, M], axis=-2)

    @abstractmethod
    def estimate_centered(
        self, src_centered: f32Array, dst_centered: f32Array
    ) -> f32Array:
        """Estimate the 2x2 linear part from (sqrt-weighted) centered points."""
        ...

    def penalty(self, T: f32Array) -> f32Array:
        return np.zeros(T.shape[:-2], dtype=T.dtype)

    def estimate_sample(self, A: f32Array, Y: f32Array) -> f32Array:
        # Unlike AffineTransform's sample fit, hypotheses must satisfy this model's
        # constraints.
        return self.estimate(A, Y, np.ones(Y.shape[:-1], dtype=np.float32))


@dataclass
class RigidTransform(_CenteredAffineTransform):
    """3-DOF rigid transform: translation and rotation only (no scale)."""

    def estimate_centered(
        self, src_centered: f32Array, dst_centered: f32Array
    ) -> f32Array:
        return _estimate_rigid_matrix(src_centered, dst_centered)


@dataclass
class SimilarityTransform(_CenteredAffineTransform):
    """4-DOF similarity transform: translation, rotation, uniform scale."""

    def estimate_centered(
        self, src_centered: f32Array, dst_centered: f32Array
    ) -> f32Array:
        R = _estimate_rigid_matrix(src_centered, dst_centered)
        numerator = np.sum(
            (src_centered @ R) * dst_centered, axis=(-2, -1), keepdims=True
        )
        denominator = np.sum(src_centered**2, axis=(-2, -1), keepdims=True)
        scale = numerator / np.maximum(denominator, 1e-8)
        return cast(f32Array, scale * R)


def _estimate_rigid_matrix(src_centered: f32Array, dst_centered: f32Array) -> f32Array:
    return _nearest_rotation(src_centered.swapaxes(-2, -1) @ dst_centered)


def _nearest_rotation(M: NDArray[np.floating]) -> f32Array:
    """Closest proper rotation (det = +1) to each matrix in M (Frobenius norm)."""
    U, _, Vt = np.linalg.svd(M)
    U[..., :, -1] *= np.sign(np.linalg.det(U @ Vt))[..., None]
    return U @ Vt


@dataclass
class AnisotropicSimilarityTransform(_CenteredAffineTransform):
    """5-DOF transform: translation, rotation, and two independent axis scales.

    The scales run along either the template's or the image's axes.
    """

    def estimate_centered(
        self, src_centered: f32Array, dst_centered: f32Array
    ) -> f32Array:
        # Fit both scale orders: template-axis scales then rotation (M = D R), or
        # rotation then image-axis scales (M = R D). With row vectors, the
        # ridge-augmented objective is tr(M^T S M) - 2 tr(M^T B) + const.
        # Prefer an orientation-preserving optimum, then the lower objective.
        # If neither candidate preserves orientation, `fit` reports infinite loss.
        # We do not search within an order for a proper alternative to its
        # mirrored optimum.
        src, dst = src_centered.astype(np.float64), dst_centered.astype(np.float64)
        S = src.swapaxes(-2, -1) @ src + _CONDITIONING_RIDGE * np.eye(2)
        B = src.swapaxes(-2, -1) @ dst

        candidates = np.stack([_scale_then_rotate(S, B), _rotate_then_scale(S, B)])
        quadratic = np.einsum("...ji,...jk,...ki->...", candidates, S, candidates)
        errors = quadratic - 2 * np.einsum("...ji,...ji->...", candidates, B)
        proper = np.linalg.det(candidates) > 0

        tie = proper[0] == proper[1]
        use_first = (proper[0] > proper[1]) | (tie & (errors[0] <= errors[1]))
        best = np.where(use_first[..., None, None], candidates[0], candidates[1])
        return best.astype(np.float32)


def _scale_then_rotate(
    S: NDArray[np.float64], B: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Least-squares M = D R (orthogonal rows), in closed form.

    Rotations preserve length, so |x D R - y| = |x D - y R^T|. For a fixed R each scale
    is a 1-D least-squares fit, which leaves a quadratic form in (cos, sin) to maximize:
    the top eigenvector of a 2x2 matrix.
    """
    variances = np.diagonal(S, axis1=-2, axis2=-1)
    # Row i holds the (cos, sin) coefficients of axis i's covariance with y R^T.
    rows = np.stack(
        [B[..., 0, :], np.stack([B[..., 1, 1], -B[..., 1, 0]], axis=-1)], axis=-2
    )
    Q = rows.swapaxes(-2, -1) @ (rows / variances[..., None])

    cos_sin = np.linalg.eigh(Q)[1][..., :, -1]
    c, s = cos_sin[..., 0], cos_sin[..., 1]
    R = np.stack([np.stack([c, s], axis=-1), np.stack([-s, c], axis=-1)], axis=-2)
    scales = (rows @ cos_sin[..., None]) / variances[..., None]
    return scales * R


def _rotate_then_scale(
    S: NDArray[np.float64], B: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Least-squares M = R D (orthogonal columns m1, m2), by bisection.

    With multiplier mu for m1 . m2 = 0, define p = m1 + m2 and q = m1 - m2.
    Stationarity gives (S + mu I) p = b1 + b2 and (S - mu I) q = b1 - b2.
    The constraint is |p| = |q|.

    For |mu| <= the smallest eigenvalue of S, the Lagrangian is convex.
    The residual |p|^2 - |q|^2 decreases with mu on this interval, so a zero
    gives the global optimum.
    """
    eigenvalues, U = np.linalg.eigh(S)

    def solve(
        rhs: NDArray[np.float64], shift: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # (S + shift I)^-1 rhs in S's eigenbasis. A singular component is left at 0 and
        # set from the constraint below.
        diagonal = eigenvalues + shift[..., None]
        return np.divide(rhs, diagonal, out=np.zeros_like(rhs), where=diagonal != 0)

    a = (U.swapaxes(-2, -1) @ (B[..., :, 0] + B[..., :, 1])[..., None])[..., 0]
    c = (U.swapaxes(-2, -1) @ (B[..., :, 0] - B[..., :, 1])[..., None])[..., 0]

    lo, hi = -eigenvalues[..., 0], eigenvalues[..., 0]
    # float64 bisection reaches machine precision well within 64 halvings.
    for _ in range(64):
        mu = (lo + hi) / 2
        positive = (solve(a, mu) ** 2 - solve(c, -mu) ** 2).sum(axis=-1) > 0
        lo, hi = np.where(positive, mu, lo), np.where(positive, hi, mu)

    # Near a multiplier bound, the weakest-axis component of q (mu > 0)
    # or p (mu < 0) is ill-conditioned. At the bound it can be undetermined
    # (the hard case). Recover it from |p| = |q| instead of dividing.
    # Use lo when growing q and hi when growing p; this leaves a nonnegative
    # squared length for the missing component, even if the midpoint rounds badly.
    upper = lo + hi >= 0
    mu = np.where(upper, lo, hi)
    p, q = solve(a, mu), solve(c, -mu)
    q0 = np.copysign(
        np.sqrt(np.maximum((p**2).sum(axis=-1) - q[..., 1] ** 2, 0)), c[..., 0]
    )
    p0 = np.copysign(
        np.sqrt(np.maximum((q**2).sum(axis=-1) - p[..., 1] ** 2, 0)), a[..., 0]
    )
    q[..., 0] = np.where(upper, q0, q[..., 0])
    p[..., 0] = np.where(upper, p[..., 0], p0)

    p, q = (U @ p[..., None])[..., 0], (U @ q[..., None])[..., 0]
    return np.stack([p + q, p - q], axis=-1) / 2


@dataclass
class BernsteinTransform(_LinearTransform):
    """Transform using Bernstein polynomial basis. Numerically stable on [0,1]^2."""

    degrees: int = 2

    @property
    def min_samples(self) -> int:
        return (self.degrees + 1) ** 2

    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        Bx = _bernstein_basis(X[..., 0], self.degrees)
        By = _bernstein_basis(X[..., 1], self.degrees)
        return (By[..., :, None] * Bx[..., None, :]).reshape(*X.shape[:-1], -1)

    def similarity_basis(self) -> f32Array:
        # Bernstein bases reproduce linear maps: the identity's control points are the
        # uniform grid (i/d, j/d).
        steps = np.linspace(0, 1, self.degrees + 1)
        grid = np.stack(np.meshgrid(steps, steps), axis=-1).reshape(-1, 2)
        return _similarity_basis(identity=grid, translation=np.ones(len(grid)))

    def warp(self, image: u8Array, template_shape: tuple[int, int]) -> u8Array:
        height, width = template_shape
        template_max_size = np.float32(max(template_shape))
        By = _bernstein_basis(
            np.arange(height, dtype=np.float32) / template_max_size, self.degrees
        )
        Bx = _bernstein_basis(
            np.arange(width, dtype=np.float32) / template_max_size, self.degrees
        )
        # The basis is a tensor product (rows of T ordered y-major), so each coordinate
        # map is By @ T_c @ Bx.T.
        n = self.degrees + 1
        T = self.matrix_.reshape(n, n, 2).transpose(2, 0, 1)
        return _remap(image, By @ T @ Bx.T)


def _bernstein_basis(t: f32Array, degrees: int) -> f32Array:
    """Evaluate the `degrees + 1` Bernstein basis polynomials at t, on a new axis."""
    k = np.arange(degrees + 1)
    binomials = np.array([math.comb(degrees, i) for i in k], dtype=np.float32)
    t = t[..., None]
    return (binomials * t**k * (1 - t) ** (degrees - k)).astype(np.float32)


def _remap(image: u8Array, maps: NDArray[np.floating]) -> u8Array:
    """Sample `image` at `maps`: (x, y) coordinates normalized by its longest side."""
    maps = maps * np.float32(max(image.shape[:2]))
    return cast(
        u8Array, cv2.remap(image, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
    )


@dataclass
class PerspectiveTransform(Transform):
    """8-DOF homography/perspective transform. Maps lines to lines."""

    @property
    def min_samples(self) -> int:
        return 4

    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        A = np.zeros((*X.shape[:-1], 2, 9), dtype=X.dtype)

        A[..., 0, :2] = X
        A[..., 0, 2] = 1
        A[..., 0, 6:] = A[..., 0, :3] * -Y[..., :1]
        A[..., 1, 3:6] = A[..., 0, :3]
        A[..., 1, 6:] = A[..., 0, :3] * -Y[..., 1:]

        return A

    def estimate(self, A: f32Array, Y: f32Array, weights: f32Array) -> f32Array:
        WA = (A * np.sqrt(weights)[..., None, None]).reshape(*A.shape[:-3], -1, 9)
        # The reduced SVD skips the large unused U, but a 4-point sample's 8 rows only
        # give its null vector in full.
        _, _, Vh = np.linalg.svd(WA, full_matrices=WA.shape[-2] < 9)
        H = Vh[..., -1, :].reshape(*A.shape[:-3], 3, 3)
        H /= H[..., -1:, -1:]
        return H.swapaxes(-2, -1)

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        Y_hat = A[..., 0, :3] @ T
        w = Y_hat[..., -1:]
        w = np.where(w == 0, np.finfo(w.dtype).eps, w)
        return (Y_hat[..., :-1] / w).astype(np.float32)

    def _preserves_orientation(self, X: f32Array, T: f32Array) -> bool:
        # The Jacobian determinant is det(H) / w**3 for projective depth w,
        # which is affine in the point. So it is positive over the bounding box exactly
        # when w has the sign of det(H) at the box's corners.
        (x0, y0), (x1, y1) = X.min(axis=0), X.max(axis=0)
        corners = np.array(
            [[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=np.float32
        )
        return bool(np.all(corners @ T[:, 2] * np.linalg.det(T) > 0))

    def penalty(self, T: f32Array) -> f32Array:
        # T is H transposed (normalized so H[2, 2] = 1): T[:2, :2] is the linear part,
        # T[:2, 2] the perspective terms.
        M = T[..., :2, :2]
        shear_and_stretch = (
            (M[..., 0, 0] - M[..., 1, 1]) ** 2 + (M[..., 0, 1] + M[..., 1, 0]) ** 2
        ) / 2
        return cast(f32Array, shear_and_stretch + (T[..., :2, 2] ** 2).sum(axis=-1))

    def warp(self, image: u8Array, template_shape: tuple[int, int]) -> u8Array:
        T = self.matrix_.copy()
        image_max_size = max(image.shape[:2])
        template_max_size = max(template_shape)
        T[:2, :2] *= image_max_size / template_max_size
        T[2, :2] *= image_max_size
        T[:2, 2] /= template_max_size
        return cast(
            u8Array,
            cv2.warpPerspective(
                image,
                T.T,
                dsize=template_shape[::-1],
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            ),
        )
