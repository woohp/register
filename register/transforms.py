from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import (
    generate_tps_remap_maps,
    get_distance_squared,
    get_pairwise_distances_squared,
)
from .utils.sampling import get_random_samples

f32Array = NDArray[np.float32]
u8Array = NDArray[np.uint8]
i64Array = NDArray[np.int64]

# Tiny ridge that keeps degenerate RANSAC samples (e.g. repeated points) solvable; it is not a prior.
_CONDITIONING_RIDGE = 1e-6


@dataclass
class Transform(ABC):
    """Base class for geometric transforms with RANSAC + IRLS fitting.

    `reg` weights `penalty`, which measures distortion beyond a similarity (shear, stretch, bending). Rotation,
    uniform scale and translation are never penalized.
    """

    reg: float = 0.005
    sigma: float = 0.00003
    n_iters_ransac: int = 4_000
    n_iters_irls: int = 50
    _matrix: f32Array | None = field(default=None, init=False, repr=False, compare=False)

    @property
    @abstractmethod
    def min_samples(self) -> int:
        """Minimum points needed to estimate transform."""
        ...

    @abstractmethod
    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        """Build design matrix A from source points X."""
        ...

    @abstractmethod
    def estimate(self, A: f32Array, Y: f32Array, weights: f32Array) -> f32Array:
        """Estimate transform parameters from design matrix A and target points Y by weighted least squares.

        `weights` has shape Y.shape[:-1] and scales each point's squared residual.
        """
        ...

    def estimate_sample(self, A: f32Array, Y: f32Array) -> f32Array:
        """Estimate RANSAC hypotheses from (batched) minimal samples. Their penalty is applied when scoring them."""
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
        """Warp `image` into the frame of a template with shape `template_shape` using the fitted transform."""
        ...

    def select(self, A: f32Array, Y: f32Array, samples: i64Array) -> tuple[f32Array, f32Array]:
        """Select subset of points for RANSAC sampling."""
        return A[samples], Y[samples]

    @property
    def matrix(self) -> f32Array:
        """Fitted transform parameters. Raises if fit() not called."""
        if self._matrix is None:
            raise ValueError("Transform not fitted")
        return self._matrix

    def fit(self, X: f32Array, Y: f32Array, weights: f32Array) -> float:
        """Fit transform from source X to target Y with point weights. Returns final loss.

        Correspondences must be sorted by decreasing match quality: RANSAC draws its early hypotheses from a
        growing prefix of the points (PROSAC-style), so unsorted input silently degrades the fit.
        """
        A = self.expand(X, Y)
        T, loss = self._ransac(A, Y, weights)
        T, loss = self._irls(A, Y, weights, T, loss)
        self._matrix = T
        return loss

    def _geman_mcclure_loss(self, d: f32Array) -> f32Array:
        return (d / (d + self.sigma)).astype(np.float32)

    def _ransac(self, A: f32Array, Y: f32Array, weights: f32Array) -> tuple[f32Array, float]:
        samples = get_random_samples(Y.shape[0], self.min_samples, self.n_iters_ransac, random.Random(4321))

        b_A, b_Y = self.select(A, Y, samples)
        Ts = self.estimate_sample(b_A, b_Y)

        distance_squared = get_distance_squared(self.transform_points(A, Ts), Y)
        losses = self._loss(distance_squared, weights, Ts)

        best_idx = np.argmin(losses)
        return Ts[best_idx], losses[best_idx].item()

    def _irls(self, A: f32Array, Y: f32Array, weights: f32Array, T: f32Array, loss: float) -> tuple[f32Array, float]:
        normalized_weights = weights / weights.sum()
        distance_squared = get_distance_squared(self.transform_points(A, T), Y)

        for _ in range(self.n_iters_irls):
            # Weighting each squared residual by the derivative of the Geman-McClure loss gives a quadratic
            # majorizer of the robust objective, so the weighted least-squares fit is the next iterate.
            irls_weights = normalized_weights * self.sigma / (distance_squared + self.sigma) ** 2
            new_T = self.estimate(A, Y, irls_weights)
            new_distance_squared = get_distance_squared(self.transform_points(A, new_T), Y)
            new_loss = self._loss(new_distance_squared, weights, new_T).item()

            if new_loss >= loss:
                break

            converged = loss - new_loss < 1e-6
            T, loss, distance_squared = new_T, new_loss, new_distance_squared
            if converged:
                break

        return T, loss

    def _loss(self, distance_squared: f32Array, weights: f32Array, T: f32Array) -> NDArray[np.floating]:
        objective = np.average(self._geman_mcclure_loss(distance_squared), axis=-1, weights=weights)
        return objective + self.reg * self.penalty(T)


@dataclass
class _LinearTransform(Transform):
    """Internal base for transforms that are linear in their parameters (Y = A @ T).

    The penalty is the squared distance of T from the subspace of similarity transforms. It is quadratic, so
    `estimate` minimizes weighted squared error + reg * penalty in closed form, matching `_loss`.
    """

    @abstractmethod
    def similarity_basis(self) -> f32Array:
        """Parameters of the x/y translations, the identity, and the 90° rotation; shape (4, *T.shape)."""
        ...

    def estimate(self, A: f32Array, Y: f32Array, weights: f32Array) -> f32Array:
        n_params = A.shape[-1]
        A_T_W = A.swapaxes(-2, -1) * weights[..., None, :]
        # Solve for vec(T) (row-major): squared error treats output columns separately, the penalty couples them.
        gram = ((A_T_W @ A)[..., :, None, :, None] * np.eye(2, dtype=A.dtype)[:, None, :]).reshape(
            *A.shape[:-2], 2 * n_params, 2 * n_params
        )
        lhs = gram + self.reg * self._non_similarity_projector() + _CONDITIONING_RIDGE * np.eye(2 * n_params)
        rhs = (A_T_W @ Y).reshape(*A.shape[:-2], 2 * n_params, 1)
        return np.linalg.solve(lhs, rhs).reshape(*A.shape[:-2], n_params, 2).astype(np.float32)

    def estimate_sample(self, A: f32Array, Y: f32Array) -> f32Array:
        # Fit the sample without the similarity prior (only the conditioning ridge) rather than pre-softening it, so
        # an extreme sample yields an extreme hypothesis for scoring to reject. This also decouples the output columns.
        A_T = A.swapaxes(-2, -1)
        lhs = A_T @ A + _CONDITIONING_RIDGE * np.eye(A.shape[-1], dtype=A.dtype)
        return cast(f32Array, np.linalg.solve(lhs, A_T @ Y))

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        return A @ T

    def penalty(self, T: f32Array) -> f32Array:
        vec_T = T.reshape(*T.shape[:-2], -1)
        return np.einsum("...i,ij,...j->...", vec_T, self._non_similarity_projector(), vec_T)

    def _non_similarity_projector(self) -> f32Array:
        basis = self.similarity_basis().reshape(4, -1).T.astype(np.float64)
        return (np.eye(basis.shape[0]) - basis @ np.linalg.pinv(basis)).astype(np.float32)


@dataclass
class AffineTransform(_LinearTransform):
    """6-DOF affine transform: translation, rotation, scale, shear."""

    @property
    def min_samples(self) -> int:
        return 3

    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        return np.pad(X, [(0, 0)] * (X.ndim - 1) + [(1, 0)], constant_values=1)

    def similarity_basis(self) -> f32Array:
        return _similarity_basis(identity=np.array([[0, 0], [1, 0], [0, 1]]), translation=np.array([1, 0, 0]))

    def warp(self, image: u8Array, template_shape: tuple[int, int]) -> u8Array:
        T = self.matrix.copy()
        image_max_size = max(image.shape[:2])
        template_max_size = max(template_shape)
        T[0] *= image_max_size
        T[1:] *= image_max_size / template_max_size
        M = np.roll(T, -1, axis=0).T
        return cast(
            u8Array, cv2.warpAffine(image, M, dsize=template_shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        )


def _similarity_basis(identity: NDArray[np.floating], translation: NDArray[np.floating]) -> f32Array:
    """Stack the x/y translations, `identity`, and `identity` rotated by 90° into a (4, K, 2) basis."""
    zeros = np.zeros_like(translation)
    x_shift = np.stack([translation, zeros], axis=-1)
    y_shift = np.stack([zeros, translation], axis=-1)
    rotated = identity @ np.array([[0, 1], [-1, 0]])
    return np.stack([x_shift, y_shift, identity, rotated]).astype(np.float32)


@dataclass
class _CenteredAffineTransform(AffineTransform):
    """Internal base for affine-like transforms estimated around source/destination centroids.

    These models are constraints in themselves (rigid and similarity cannot shear, and stretch is the point of the
    anisotropic model), so they carry no penalty.
    """

    def estimate(self, A: f32Array, Y: f32Array, weights: f32Array) -> f32Array:
        src_xy = A[..., 1:]
        w = weights[..., None]

        src_center = (w * src_xy).sum(axis=-2, keepdims=True) / w.sum(axis=-2, keepdims=True)
        dst_center = (w * Y).sum(axis=-2, keepdims=True) / w.sum(axis=-2, keepdims=True)

        # Scaling centered points by sqrt(w) turns every unweighted moment below into its weighted counterpart.
        sqrt_w = np.sqrt(w)
        M = self.estimate_centered(sqrt_w * (src_xy - src_center), sqrt_w * (Y - dst_center))
        t = dst_center - src_center @ M

        return np.concatenate([t, M], axis=-2)

    @abstractmethod
    def estimate_centered(self, src_centered: f32Array, dst_centered: f32Array) -> f32Array:
        """Estimate the linear 2x2 component from (sqrt-weighted) centered source/destination points."""
        ...

    def penalty(self, T: f32Array) -> f32Array:
        return np.zeros(T.shape[:-2], dtype=T.dtype)


@dataclass
class RigidTransform(_CenteredAffineTransform):
    """3-DOF rigid transform: translation and rotation only (no scale)."""

    def estimate_centered(self, src_centered: f32Array, dst_centered: f32Array) -> f32Array:
        return _estimate_rigid_matrix(src_centered, dst_centered)


@dataclass
class SimilarityTransform(_CenteredAffineTransform):
    """4-DOF similarity transform: translation, rotation, uniform scale."""

    def estimate_centered(self, src_centered: f32Array, dst_centered: f32Array) -> f32Array:
        R = _estimate_rigid_matrix(src_centered, dst_centered)
        numerator = np.sum((src_centered @ R) * dst_centered, axis=(-2, -1), keepdims=True)
        denominator = np.sum(src_centered**2, axis=(-2, -1), keepdims=True)
        scale = numerator / np.maximum(denominator, 1e-8)
        return cast(f32Array, scale * R)


@dataclass
class AnisotropicSimilarityTransform(_CenteredAffineTransform):
    """5-DOF transform: translation, rotation, and independent x/y scale."""

    def estimate_centered(self, src_centered: f32Array, dst_centered: f32Array) -> f32Array:
        SXX = src_centered.swapaxes(-2, -1) @ src_centered
        SXY = src_centered.swapaxes(-2, -1) @ dst_centered
        Id = np.eye(2, dtype=src_centered.dtype)
        M_aff = np.linalg.solve(SXX + _CONDITIONING_RIDGE * Id, SXY)

        MTM = M_aff.swapaxes(-2, -1) @ M_aff
        MMT = M_aff @ M_aff.swapaxes(-2, -1)

        off_diag_mtm = np.abs(MTM[..., 0, 1])
        off_diag_mmt = np.abs(MMT[..., 0, 1])

        scale_on_right = off_diag_mtm <= off_diag_mmt

        s_rd = np.sqrt(np.maximum(np.diagonal(MTM, axis1=-2, axis2=-1), 1e-8))
        s_dr = np.sqrt(np.maximum(np.diagonal(MMT, axis1=-2, axis2=-1), 1e-8))

        # M = R @ D: divide out the column scales; M = D @ R: divide out the row scales.
        M_rd = _nearest_rotation(M_aff / s_rd[..., None, :]) * s_rd[..., None, :]
        M_dr = s_dr[..., :, None] * _nearest_rotation(M_aff / s_dr[..., :, None])
        return np.where(scale_on_right[..., None, None], M_rd, M_dr)


def _estimate_rigid_matrix(src_centered: f32Array, dst_centered: f32Array) -> f32Array:
    return _nearest_rotation(src_centered.swapaxes(-2, -1) @ dst_centered)


def _nearest_rotation(M: NDArray[np.floating]) -> f32Array:
    """Closest proper rotation (det = +1) to each matrix in M, in the Frobenius sense."""
    U, _, Vt = np.linalg.svd(M)
    U[..., :, -1] *= np.sign(np.linalg.det(U @ Vt))[..., None]
    return U @ Vt


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
        # Bernstein bases reproduce linear maps: the identity's control points are the uniform grid (i/d, j/d).
        steps = np.linspace(0, 1, self.degrees + 1)
        grid = np.stack(np.meshgrid(steps, steps), axis=-1).reshape(-1, 2)
        return _similarity_basis(identity=grid, translation=np.ones(len(grid)))

    def warp(self, image: u8Array, template_shape: tuple[int, int]) -> u8Array:
        height, width = template_shape
        template_max_size = np.float32(max(template_shape))
        By = _bernstein_basis(np.arange(height, dtype=np.float32) / template_max_size, self.degrees)
        Bx = _bernstein_basis(np.arange(width, dtype=np.float32) / template_max_size, self.degrees)
        # The basis is a tensor product (rows of T ordered y-major), so each coordinate map is By @ T_c @ Bx.T.
        T = self.matrix.reshape(self.degrees + 1, self.degrees + 1, 2).transpose(2, 0, 1)
        return _remap(image, By @ T @ Bx.T)


def _bernstein_basis(t: f32Array, degrees: int) -> f32Array:
    """Evaluate the `degrees + 1` Bernstein basis polynomials at t, along a new trailing axis."""
    k = np.arange(degrees + 1)
    binomials = np.array([math.comb(degrees, i) for i in k], dtype=np.float32)
    t = t[..., None]
    return (binomials * t**k * (1 - t) ** (degrees - k)).astype(np.float32)


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
        WA = A * np.sqrt(weights)[..., None, None]
        _, _, Vh = np.linalg.svd(WA.reshape(*A.shape[:-3], -1, 9))
        H = Vh[..., -1, :].reshape(*A.shape[:-3], 3, 3)
        H /= H[..., -1:, -1:]
        return H.swapaxes(-2, -1)

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        Y_hat = A[..., 0, :3] @ T
        w = Y_hat[..., -1:]
        w = np.where(w == 0, np.finfo(w.dtype).eps, w)
        return (Y_hat[..., :-1] / w).astype(np.float32)

    def penalty(self, T: f32Array) -> f32Array:
        # T is H transposed (normalized so H[2, 2] = 1): T[:2, :2] is the linear part, T[:2, 2] the perspective terms.
        M = T[..., :2, :2]
        shear_and_stretch = ((M[..., 0, 0] - M[..., 1, 1]) ** 2 + (M[..., 0, 1] + M[..., 1, 0]) ** 2) / 2
        return cast(f32Array, shear_and_stretch + (T[..., :2, 2] ** 2).sum(axis=-1))

    def warp(self, image: u8Array, template_shape: tuple[int, int]) -> u8Array:
        T = self.matrix.copy()
        image_max_size = max(image.shape[:2])
        template_max_size = max(template_shape)
        T[:2, :2] *= image_max_size / template_max_size
        T[2, :2] *= image_max_size
        T[:2, 2] /= template_max_size
        return cast(
            u8Array,
            cv2.warpPerspective(image, T.T, dsize=template_shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
        )


@dataclass
class ThinPlateSplineTransform(Transform):
    """Non-rigid transform using thin-plate spline radial basis function."""

    @property
    def min_samples(self) -> int:
        return 5

    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        K = _tps_kernel(X, X)
        P = np.pad(X, [(0, 0)] * (X.ndim - 1) + [(1, 0)], constant_values=1)
        zeros = np.zeros((*X.shape[:-2], 3, 3), dtype=X.dtype)

        A_top = np.concatenate([K, P], axis=-1)
        A_bottom = np.concatenate([P.swapaxes(-1, -2), zeros], axis=-1)
        A = np.concatenate([A_top, A_bottom], axis=-2)

        return A

    def select(self, A: f32Array, Y: f32Array, samples: i64Array) -> tuple[f32Array, f32Array]:
        """Re-expand design matrix for sampled control points."""
        X = A[..., :-3, -2:]
        Y_sampled = Y[samples]
        A_sampled = self.expand(X[samples], Y_sampled)
        return A_sampled, Y_sampled

    def estimate(self, A: f32Array, Y: f32Array, weights: f32Array) -> f32Array:
        N = Y.shape[-2]
        Y_aug = np.pad(Y, [(0, 0)] * (Y.ndim - 2) + [(0, 3), (0, 0)], constant_values=0)

        # Weighted smoothing spline: a point with weight w gets smoothing reg / w.
        A_reg = A.copy()
        A_reg[..., range(N), range(N)] += self.reg / weights
        T = np.linalg.solve(A_reg, Y_aug)

        X = A[..., :-3, -2:]
        X_padded = np.pad(X, [(0, 0)] * (X.ndim - 2) + [(0, 3), (0, 0)])
        return np.concatenate([T, X_padded], axis=-1)

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        X_new = A[..., :-3, -2:]
        X_old = T[..., :-3, -2:]
        W = T[..., :-3, :2]
        ab = T[..., -3:, :2]

        K = _tps_kernel(X_new, X_old)
        P = np.pad(X_new, [(0, 0)] * (X_new.ndim - 1) + [(1, 0)], constant_values=1)
        return K @ W + P @ ab

    def penalty(self, T: f32Array) -> f32Array:
        """Bending energy penalty: trace(W.T @ K @ W)."""
        W = T[..., :-3, :2]
        X = T[..., :-3, -2:]
        K = _tps_kernel(X, X)
        return np.einsum("...id,...ij,...jd->...", W, K, W)

    def warp(self, image: u8Array, template_shape: tuple[int, int]) -> u8Array:
        return _remap(image, generate_tps_remap_maps(*template_shape, self.matrix))


def _tps_kernel(X1: f32Array, X2: f32Array) -> f32Array:
    """TPS radial basis U = r² log r² between every point of X1 and every point of X2, broadcasting batch dims."""
    batch_shape = np.broadcast_shapes(X1.shape[:-2], X2.shape[:-2])
    X1 = np.broadcast_to(X1, batch_shape + X1.shape[-2:])
    X2 = np.broadcast_to(X2, batch_shape + X2.shape[-2:])
    distance_squared = get_pairwise_distances_squared(X1, X2)
    return distance_squared * np.log(distance_squared.clip(min=1e-6))


def _remap(image: u8Array, maps: NDArray[np.floating]) -> u8Array:
    """Sample `image` at `maps`, which holds (x, y) coordinates normalized by the image's longest side."""
    maps = maps * np.float32(max(image.shape[:2]))
    return cast(u8Array, cv2.remap(image, maps[0], maps[1], interpolation=cv2.INTER_LINEAR))
