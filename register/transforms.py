from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import get_distance_squared, get_pairwise_distances_squared

f32Array = NDArray[np.float32]
u8Array = NDArray[np.uint8]
i64Array = NDArray[np.int64]


@dataclass
class Transform(ABC):
    """Base class for geometric transforms with RANSAC + IRLS fitting."""

    reg: float = 0.005
    sigma: float = 0.00003
    n_iters_ransac: int = 4_000
    n_iters_irls: int = 50
    _matrix: f32Array | None = field(default=None, repr=False)

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
    def estimate(self, A: f32Array, Y: f32Array) -> f32Array:
        """Estimate transform parameters from design matrix A and target points Y."""
        ...

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
    def warp(self, image: u8Array, template_shape: tuple[int, int], image_max_size: int) -> u8Array:
        """Warp image using fitted transform."""
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
        """Fit transform from source X to target Y with point weights. Returns final loss."""
        A = self.expand(X, Y)
        T, loss = self._ransac(A, Y, weights)
        T, loss = self._irls(A, Y, weights, T, loss)
        self._matrix = T
        return loss

    def _geman_mcclure_loss(self, d: f32Array) -> f32Array:
        return (d / (d + self.sigma)).astype(np.float32)

    def _ransac(self, A: f32Array, Y: f32Array, weights: f32Array) -> tuple[f32Array, float]:
        from .utils.sampling import get_random_samples

        samples = get_random_samples(Y.shape[0], self.min_samples, self.n_iters_ransac, random.Random(4321))

        b_A, b_Y = self.select(A, Y, samples)
        Ts = self.estimate(b_A, b_Y)

        Y_hat = self.transform_points(A, Ts)
        distance_squared = get_distance_squared(Y_hat, Y)
        objective_losses = np.average(self._geman_mcclure_loss(distance_squared), axis=-1, weights=weights)
        weights_losses = self.penalty(Ts)
        losses = objective_losses + self.reg * weights_losses

        best_idx = np.argmin(losses)
        return Ts[best_idx], losses[best_idx].item()

    def _irls(
        self, A: f32Array, Y: f32Array, weights: f32Array, T: f32Array, initial_loss: float
    ) -> tuple[f32Array, float]:
        last_T = T
        last_loss = initial_loss

        Y_hat = self.transform_points(A, T)
        distance_squared = get_distance_squared(Y_hat, Y)
        w = (weights.astype(np.float64) * self.sigma / (distance_squared.astype(np.float64) + self.sigma) ** 2).astype(
            np.float32
        )

        for _ in range(self.n_iters_irls):
            w_ext = np.ones((A.shape[0],) + w.shape[1:], dtype=w.dtype)
            w_ext[: w.shape[0]] = w
            w_reshaped = w_ext.reshape([-1] + [1] * (A.ndim - 1))
            WA = A * w_reshaped
            WY = np.pad(Y, [(0, A.shape[0] - Y.shape[0]), (0, 0)], constant_values=0) * w_reshaped
            T = self.estimate(WA, WY)

            Y_hat = self.transform_points(A, T)
            distance_squared = get_distance_squared(Y_hat, Y)
            w = (
                weights.astype(np.float64) * self.sigma / (distance_squared.astype(np.float64) + self.sigma) ** 2
            ).astype(np.float32)
            objective_loss = np.average(self._geman_mcclure_loss(distance_squared), axis=-1, weights=weights)
            weights_loss = self.penalty(T)
            loss: float = (objective_loss + weights_loss * self.reg).item()

            if loss >= last_loss:
                return last_T, last_loss
            elif last_loss - loss < 1e-6:
                return T, loss

            last_T = T
            last_loss = loss

        return T, last_loss


@dataclass
class AffineTransform(Transform):
    """6-DOF affine transform: translation, rotation, scale, shear."""

    @property
    def min_samples(self) -> int:
        return 3

    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        return np.pad(X, [(0, 0)] * (X.ndim - 1) + [(1, 0)], constant_values=1)

    def estimate(self, A: f32Array, Y: f32Array) -> f32Array:
        A_T = A.swapaxes(-2, -1)
        return np.linalg.solve(A_T @ A + self.reg * np.eye(A.shape[-1], dtype=A.dtype), A_T @ Y)

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        return A @ T

    def penalty(self, T: f32Array) -> f32Array:
        res = T[..., 1:, :].copy()
        res[..., 0, 0] -= 1
        res[..., 1, 1] -= 1
        return (res**2).sum(axis=(-2, -1))

    def warp(self, image: u8Array, template_shape: tuple[int, int], image_max_size: int) -> u8Array:
        T = self.matrix.copy()
        template_max_size = max(template_shape)
        T[0] *= image_max_size
        T[1:] *= image_max_size / template_max_size
        M = np.roll(T, -1, axis=0).T
        return cv2.warpAffine(image, M, dsize=template_shape[::-1], flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)  # type: ignore[return-value]


@dataclass
class _CenteredAffineTransform(AffineTransform):
    """Internal base for affine-like transforms estimated around source/destination centroids."""

    def estimate(self, A: f32Array, Y: f32Array) -> f32Array:
        src_xy = A[..., 1:]
        dst_xy = Y

        src_center = np.mean(src_xy, axis=-2, keepdims=True)
        dst_center = np.mean(dst_xy, axis=-2, keepdims=True)

        M = self.estimate_centered(src_xy - src_center, dst_xy - dst_center)
        t = dst_center - src_center @ M

        return np.concatenate([t, M], axis=-2)

    @abstractmethod
    def estimate_centered(self, src_centered: f32Array, dst_centered: f32Array) -> f32Array:
        """Estimate the linear 2x2 component from centered source/destination points."""
        ...


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
        return scale * R


@dataclass
class AnisotropicSimilarityTransform(_CenteredAffineTransform):
    """5-DOF transform: translation, rotation, and independent x/y scale."""

    def estimate_centered(self, src_centered: f32Array, dst_centered: f32Array) -> f32Array:
        SXX = src_centered.swapaxes(-2, -1) @ src_centered
        SXY = src_centered.swapaxes(-2, -1) @ dst_centered
        Id = np.eye(2, dtype=src_centered.dtype)
        M_aff = np.linalg.solve(SXX + (self.reg + 1e-8) * Id, SXY)

        MTM = M_aff.swapaxes(-2, -1) @ M_aff
        MMT = M_aff @ M_aff.swapaxes(-2, -1)

        off_diag_mtm = np.abs(MTM[..., 0, 1])
        off_diag_mmt = np.abs(MMT[..., 0, 1])

        scale_on_right = off_diag_mtm <= off_diag_mmt

        s_rd = np.sqrt(np.maximum(np.diagonal(MTM, axis1=-2, axis2=-1), 1e-8))
        s_dr = np.sqrt(np.maximum(np.diagonal(MMT, axis1=-2, axis2=-1), 1e-8))

        M = np.empty_like(M_aff)

        mask = scale_on_right
        if np.any(mask):
            Mi, si = M_aff[mask], s_rd[mask]
            Di_inv = 1.0 / si[..., None, :] * np.eye(2, dtype=Mi.dtype)
            Ri_est = Mi @ Di_inv
            Ui, _, Vti = np.linalg.svd(Ri_est)
            Ri = Ui @ Vti
            det_i = np.linalg.det(Ri)
            needs_flip = det_i < 0
            if np.any(needs_flip):
                Vti[needs_flip, -1, :] *= -1
                Ri[needs_flip] = Ui[needs_flip] @ Vti[needs_flip]
            M[mask] = Ri * si[..., None, :]

        mask = ~scale_on_right
        if np.any(mask):
            Mi, si = M_aff[mask], s_dr[mask]
            Di_inv = 1.0 / si[..., None, :] * np.eye(2, dtype=Mi.dtype)
            Ri_est = Di_inv.swapaxes(-2, -1) @ Mi
            Ui, _, Vti = np.linalg.svd(Ri_est)
            Ri = Ui @ Vti
            det_i = np.linalg.det(Ri)
            needs_flip = det_i < 0
            if np.any(needs_flip):
                Vti[needs_flip, -1, :] *= -1
                Ri[needs_flip] = Ui[needs_flip] @ Vti[needs_flip]
            M[mask] = si[..., :, None] * Ri

        return M


def _estimate_rigid_matrix(src_centered: f32Array, dst_centered: f32Array) -> f32Array:
    H = src_centered.swapaxes(-2, -1) @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R_rot = Vt.swapaxes(-2, -1) @ U.swapaxes(-2, -1)

    needs_inverting = np.linalg.det(R_rot) < 0
    if np.any(needs_inverting):
        Vt_copy = Vt.copy()
        Vt_copy[needs_inverting, -1, :] *= -1
        R_rot[needs_inverting] = Vt_copy[needs_inverting].swapaxes(-2, -1) @ U[needs_inverting].swapaxes(-2, -1)

    return R_rot.swapaxes(-2, -1)


@dataclass
class BernsteinTransform(Transform):
    """Transform using Bernstein polynomial basis. Numerically stable on [0,1]^2."""

    degrees: int = 2

    @property
    def min_samples(self) -> int:
        return (self.degrees + 1) ** 2

    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        x = X[..., 0:1]
        y = X[..., 1:2]
        d = self.degrees

        basis_list = []
        for j in range(d + 1):
            by = math.comb(d, j) * (y**j) * ((1 - y) ** (d - j))
            for i in range(d + 1):
                bx = math.comb(d, i) * (x**i) * ((1 - x) ** (d - i))
                basis_list.append(bx * by)

        return np.concatenate(basis_list, axis=-1)

    def estimate(self, A: f32Array, Y: f32Array) -> f32Array:
        A_T = A.swapaxes(-2, -1)
        return np.linalg.solve(A_T @ A + self.reg * np.eye(A.shape[-1], dtype=A.dtype), A_T @ Y)

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        return A @ T

    def penalty(self, T: f32Array) -> f32Array:
        return np.zeros(T.shape[:-2], dtype=T.dtype)

    def warp(self, image: u8Array, template_shape: tuple[int, int], image_max_size: int) -> u8Array:
        if self.degrees == 2:
            from .utils import generate_bernstein2_remap_maps

            map1, map2 = generate_bernstein2_remap_maps(*template_shape, self.matrix)
        else:
            from .utils import generate_bernstein_remap_maps

            map1, map2 = generate_bernstein_remap_maps(*template_shape, self.degrees, self.matrix)

        return cv2.remap(  # type: ignore[return-value]
            image,
            (map1 * image_max_size).astype(np.float32),
            (map2 * image_max_size).astype(np.float32),
            interpolation=cv2.INTER_AREA,
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

    def estimate(self, A: f32Array, Y: f32Array) -> f32Array:
        _, _, V = np.linalg.svd(A.reshape(A.shape[:-3] + (-1, 9)))
        Ts = V[:, -1].reshape(-1, 3, 3)
        Ts /= Ts[:, -1:, -1:]
        Ts = Ts.swapaxes(1, 2)
        return Ts

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        Y_hat = A[..., 0, :3] @ T
        Y_hat[Y_hat == 0] = np.finfo(Y_hat.dtype).eps
        Y_hat = Y_hat[..., :-1] / Y_hat[..., -1:]
        return Y_hat.astype(np.float32)

    def penalty(self, T: f32Array) -> f32Array:
        res = T.copy()
        res[..., 0, 0] -= 1
        res[..., 1, 1] -= 1
        res[..., 2, 2] -= 1
        return (res**2).sum(axis=(-2, -1))

    def warp(self, image: u8Array, template_shape: tuple[int, int], image_max_size: int) -> u8Array:
        T = self.matrix.copy()
        template_max_size = max(template_shape)
        T[:2, :2] *= image_max_size / template_max_size
        T[2, :2] *= image_max_size
        T[:2, 2] /= template_max_size
        return cv2.warpPerspective(image, T.T, dsize=template_shape[::-1], flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)  # type: ignore[return-value]


@dataclass
class ThinPlateSplineTransform(Transform):
    """Non-rigid transform using thin-plate spline radial basis function."""

    @property
    def min_samples(self) -> int:
        return 5

    def expand(self, X: f32Array, Y: f32Array) -> f32Array:
        pairwise_distance_squared = get_pairwise_distances_squared(X.astype(np.float32), X.astype(np.float32))
        K = pairwise_distance_squared * np.log(pairwise_distance_squared.clip(min=1e-6))

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

    def estimate(self, A: f32Array, Y: f32Array) -> f32Array:
        N = Y.shape[-2]
        if Y.shape[-2] != A.shape[-1]:
            Y_aug = np.pad(Y, [(0, 0)] * (Y.ndim - 2) + [(0, 3), (0, 0)], constant_values=0)
        else:
            Y_aug = Y

        if self.reg > 0:
            A_reg = A.copy()
            A_reg[..., :N, :N] += self.reg * np.eye(N, dtype=A.dtype)
            T = np.linalg.solve(A_reg, Y_aug)
        else:
            T = np.linalg.solve(A, Y_aug)

        X = A[..., :-3, -2:]
        X_padded = np.pad(X, [(0, 0)] * (X.ndim - 2) + [(0, 3), (0, 0)])
        return np.concatenate([T, X_padded], axis=-1)

    @staticmethod
    def transform_points(A: f32Array, T: f32Array) -> f32Array:
        X_new = A[..., :-3, -2:]
        X_old = T[..., :-3, -2:]
        W = T[..., :-3, :2]
        ab = T[..., -3:, :2]

        if T.ndim == 3:
            X_new_batch = X_new[None].repeat(T.shape[0], axis=0) if X_new.ndim == 2 else X_new
            pairwise_distance_squared = get_pairwise_distances_squared(
                X_new_batch.astype(np.float32), X_old.astype(np.float32)
            )
        else:
            pairwise_distance_squared = get_pairwise_distances_squared(
                X_new.astype(np.float32), X_old.astype(np.float32)
            )

        K = pairwise_distance_squared * np.log(pairwise_distance_squared.clip(min=1e-6))
        P = np.pad(X_new, [(0, 0)] * (X_new.ndim - 1) + [(1, 0)], constant_values=1)

        if T.ndim == 3:
            P_for_matmul = P[None] if P.ndim == 2 else P
            return ((K @ W) + (P_for_matmul @ ab)).astype(np.float32)
        else:
            return ((K @ W) + (P @ ab)).astype(np.float32)

    def penalty(self, T: f32Array) -> f32Array:
        """Bending energy penalty: trace(W.T @ K @ W)."""
        W = T[..., :-3, :2]
        X = T[..., :-3, -2:]
        pairwise_distance_squared = get_pairwise_distances_squared(X.astype(np.float32), X.astype(np.float32))
        K = pairwise_distance_squared * np.log(pairwise_distance_squared.clip(min=1e-6))
        return np.einsum("...id,...ij,...jd->...", W, K, W)

    def warp(self, image: u8Array, template_shape: tuple[int, int], image_max_size: int) -> u8Array:
        from .utils import generate_tps_remap_maps

        map1, map2 = generate_tps_remap_maps(*template_shape, self.matrix)
        map1 = (map1 * image_max_size).astype(np.float32)
        map2 = (map2 * image_max_size).astype(np.float32)
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_AREA)  # type: ignore[return-value]
