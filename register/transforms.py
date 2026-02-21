from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from .utils import get_pairwise_distances_squared

FloatingNDArray = TypeVar("FloatingNDArray", NDArray[np.float32], NDArray[np.float64])
method_type = Literal["perspective", "affine", "rigid", "similarity", "quadratic", "thin_plate_spline"]


class Transform(ABC):
    min_samples: int

    @staticmethod
    @abstractmethod
    def expand(X: FloatingNDArray, Y: FloatingNDArray) -> FloatingNDArray:
        """Expand the data into their bases according to the basis function"""

    @staticmethod
    def select(
        A: FloatingNDArray, Y: FloatingNDArray, samples: NDArray[np.int64]
    ) -> tuple[FloatingNDArray, FloatingNDArray]:
        return A[samples], Y[samples]

    @staticmethod
    @abstractmethod
    def estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float) -> FloatingNDArray:
        pass

    @staticmethod
    @abstractmethod
    def transform(A: FloatingNDArray, T: FloatingNDArray) -> FloatingNDArray:
        pass

    @staticmethod
    def get_transform(method: method_type) -> type[Transform]:
        match method:
            case "perspective":
                return PerspectiveTransform
            case "affine":
                return AffineTransform
            case "rigid":
                return RigidTransform
            case "similarity":
                return SimilarityTransform
            case "quadratic":
                return QuadraticTransform
            case "thin_plate_spline":
                return ThinPlateSplineTransform
            case _:
                raise ValueError(f"unrecognized method: {method_type!r}")


def poly_expand(X: FloatingNDArray, d: int) -> FloatingNDArray:
    if d == 1:
        return np.pad(X, [(0, 0)] * (X.ndim - 1) + [(1, 0)], constant_values=1)

    o = np.ones((*X.shape[:-1], 1), dtype=X.dtype)

    if d == 2:
        x, y = np.split(X, 2, axis=-1)
        xx = np.square(x)
        yy = np.square(y)
        xy = x * y
        return np.concatenate((o, X, xx, xy, yy), axis=-1)
    else:
        raise NotImplementedError


class PolynomialTransform(Transform):
    @staticmethod
    def estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float) -> FloatingNDArray:
        # A: [..., samples, K] where K is determined by the degree
        # Y: [..., samples, 2]
        # solve regularized least squares
        # https://stackoverflow.com/questions/27476933/numpy-linear-regression-with-regularization
        A_T = A.swapaxes(-2, -1)
        res = np.linalg.solve(A_T @ A + lamb * np.eye(A.shape[-1], dtype=A.dtype), A_T @ Y)
        return cast(FloatingNDArray, res)

    @staticmethod
    def transform(A: FloatingNDArray, T: FloatingNDArray) -> FloatingNDArray:
        return A @ T


class AffineTransform(PolynomialTransform):
    min_samples = 3

    @staticmethod
    def expand(X: FloatingNDArray, Y: FloatingNDArray) -> FloatingNDArray:
        return poly_expand(X, 1)


class SimilarityTransform(AffineTransform):
    """A similarity transform is a subgroup of the affine transform that allows for translation, rotation, uniform
    scaling, and reflection.

    See:
        https://en.wikipedia.org/wiki/Similarity_(geometry)
        https://math.stackexchange.com/a/3789053
        https://ecommons.udayton.edu/cgi/viewcontent.cgi?article=1173&context=mee_fac_pub
        https://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf

    """

    @staticmethod
    def _estimate_rigid(src_centered: FloatingNDArray, dst_centered: FloatingNDArray) -> FloatingNDArray:
        # 3-DOF Rigid estimation (SVD)
        H = src_centered.swapaxes(-2, -1) @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R_rot = Vt.swapaxes(-2, -1) @ U.swapaxes(-2, -1)

        needs_inverting = np.linalg.det(R_rot) < 0
        if np.any(needs_inverting):
            Vt_copy = Vt.copy()
            Vt_copy[needs_inverting, -1, :] *= -1
            R_rot[needs_inverting] = Vt_copy[needs_inverting].swapaxes(-2, -1) @ U[needs_inverting].swapaxes(-2, -1)

        return R_rot.swapaxes(-2, -1)

    @staticmethod
    def _estimate_anisotropic(
        src_centered: FloatingNDArray, dst_centered: FloatingNDArray, lamb: float
    ) -> FloatingNDArray:
        # 5-DOF Anisotropic Similarity: Solve for best-fit RD or DR matrix with 0 shear.
        SXX = src_centered.swapaxes(-2, -1) @ src_centered
        SXY = src_centered.swapaxes(-2, -1) @ dst_centered
        Id = np.identity(2, dtype=src_centered.dtype)
        M_aff = np.linalg.solve(SXX + (lamb + 1e-8) * Id, SXY)

        # Check which model has less shear: Rotate-then-Scale (RD) or Scale-then-Rotate (DR)
        MTM = M_aff.swapaxes(-2, -1) @ M_aff
        MMT = M_aff @ M_aff.swapaxes(-2, -1)

        # Off-diagonals represent shear in that coordinate system
        off_diag_mtm = np.abs(MTM[..., 0, 1])
        off_diag_mmt = np.abs(MMT[..., 0, 1])

        # scale_on_right means M = R @ D
        scale_on_right = off_diag_mtm <= off_diag_mmt

        # Extract scales from diagonals (preserves x/y axis mapping)
        s_rd = np.sqrt(np.maximum(np.diagonal(MTM, axis1=-2, axis2=-1), 1e-8))
        s_dr = np.sqrt(np.maximum(np.diagonal(MMT, axis1=-2, axis2=-1), 1e-8))

        # Vectorized selection and decomposition
        M = np.empty_like(M_aff)

        # Case 1: Rotate-then-Scale (M = R @ D)
        mask = scale_on_right
        if np.any(mask):
            Mi, si = M_aff[mask], s_rd[mask]
            Di_inv = 1.0 / si[..., None, :] * np.identity(2, dtype=Mi.dtype)
            Ri_est = Mi @ Di_inv
            Ui, _, Vti = np.linalg.svd(Ri_est)
            Ri = Ui @ Vti
            det_i = np.linalg.det(Ri)
            needs_flip = det_i < 0
            if np.any(needs_flip):
                Vti[needs_flip, -1, :] *= -1
                Ri[needs_flip] = Ui[needs_flip] @ Vti[needs_flip]
            M[mask] = Ri * si[..., None, :]

        # Case 2: Scale-then-Rotate (M = D @ R)
        mask = ~scale_on_right
        if np.any(mask):
            Mi, si = M_aff[mask], s_dr[mask]
            Di_inv = 1.0 / si[..., None, :] * np.identity(2, dtype=Mi.dtype)
            Ri_est = Di_inv.swapaxes(-2, -1) @ Mi  # D^-1 is diagonal, so swap is just for matmul
            Ui, _, Vti = np.linalg.svd(Ri_est)
            Ri = Ui @ Vti
            det_i = np.linalg.det(Ri)
            needs_flip = det_i < 0
            if np.any(needs_flip):
                Vti[needs_flip, -1, :] *= -1
                Ri[needs_flip] = Ui[needs_flip] @ Vti[needs_flip]
            M[mask] = si[..., :, None] * Ri

        return M

    @staticmethod
    def _estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float, compute_scale: bool) -> FloatingNDArray:
        src_xy = A[..., 1:]
        dst_xy = Y

        src_center = np.mean(src_xy, axis=-2, keepdims=True)
        dst_center = np.mean(dst_xy, axis=-2, keepdims=True)

        src_centered = src_xy - src_center
        dst_centered = dst_xy - dst_center

        if compute_scale:
            M = SimilarityTransform._estimate_anisotropic(src_centered, dst_centered, lamb)
        else:
            M = SimilarityTransform._estimate_rigid(src_centered, dst_centered)

        t = dst_center - src_center @ M

        return np.concatenate([t, M], axis=-2)

    @staticmethod
    def estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float) -> FloatingNDArray:
        return SimilarityTransform._estimate(A, Y, lamb, compute_scale=True)


class RigidTransform(SimilarityTransform):
    @staticmethod
    def estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float) -> FloatingNDArray:
        return SimilarityTransform._estimate(A, Y, lamb, compute_scale=False)


class QuadraticTransform(PolynomialTransform):
    min_samples = 6

    @staticmethod
    def expand(X: FloatingNDArray, Y: FloatingNDArray) -> FloatingNDArray:
        return poly_expand(X, 2)


class PerspectiveTransform(Transform):
    min_samples = 4

    @staticmethod
    def expand(X: FloatingNDArray, Y: FloatingNDArray) -> FloatingNDArray:
        # X: [..., 2]
        # Y: [..., 2]
        A = np.zeros((*X.shape[:-1], 2, 9), dtype=X.dtype)

        A[..., 0, :2] = X
        A[..., 0, 2] = 1
        A[..., 0, 6:] = A[..., 0, :3] * -Y[..., :1]
        A[..., 1, 3:6] = A[..., 0, :3]
        A[..., 1, 6:] = A[..., 0, :3] * -Y[..., 1:]

        return A

    @staticmethod
    def estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float) -> FloatingNDArray:
        # A: [..., 2, 9]
        # Y: [..., 2], not used
        _, _, V = np.linalg.svd(A.reshape(A.shape[:-3] + (-1, 9)))
        Ts = V[:, -1].reshape(-1, 3, 3)
        Ts /= Ts[:, -1:, -1:]
        Ts = Ts.swapaxes(1, 2)
        return Ts

    @staticmethod
    def transform(A: FloatingNDArray, T: FloatingNDArray) -> FloatingNDArray:
        Y_hat = A[..., 0, :3] @ T
        Y_hat[Y_hat == 0] = np.finfo(Y_hat.dtype).eps
        Y_hat = Y_hat[..., :-1] / Y_hat[..., -1:]
        return Y_hat


class ThinPlateSplineTransform(Transform):
    min_samples = 5

    @staticmethod
    def expand(X: FloatingNDArray, Y: FloatingNDArray) -> FloatingNDArray:
        # X: [..., N, 2]
        pairwise_distance_squared = get_pairwise_distances_squared(X.astype(np.float32), X.astype(np.float32))
        K = pairwise_distance_squared * np.log(pairwise_distance_squared.clip(min=1e-6))

        P = np.pad(X, [(0, 0)] * (X.ndim - 1) + [(1, 0)], constant_values=1)
        zeros = np.zeros((*X.shape[:-2], 3, 3), dtype=X.dtype)

        A_top = np.concatenate([K, P], axis=-1)
        A_bottom = np.concatenate([P.swapaxes(-1, -2), zeros], axis=-1)
        A = np.concatenate([A_top, A_bottom], axis=-2)

        return A

    @staticmethod
    def select(
        A: FloatingNDArray, Y: FloatingNDArray, samples: NDArray[np.int64]
    ) -> tuple[FloatingNDArray, FloatingNDArray]:
        # A is [N+3, N+3], Y is [N, 2]
        # We need to extract X from A or re-expand
        # A was returned by expand(X, Y), so K = A[..., :-3, :-3], P = A[..., :-3, -3:]
        # Actually, it's easier to just pass X around or reconstruct it from P
        X = A[..., :-3, -2:]
        Y_sampled = Y[samples]
        A_sampled = ThinPlateSplineTransform.expand(X[samples], Y_sampled)
        return A_sampled, Y_sampled

    @staticmethod
    def estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float) -> FloatingNDArray:
        if Y.shape[-2] != A.shape[-1]:
            Y_aug = np.pad(Y, [(0, 0)] * (Y.ndim - 2) + [(0, 3), (0, 0)], constant_values=0)
        else:
            Y_aug = Y
        T = np.linalg.solve(A + lamb * np.identity(A.shape[-1], dtype=A.dtype), Y_aug)
        X = A[..., :-3, -2:]
        # T: [..., N+3, 2]
        # We append X so that transform() knows the control points
        X_padded = np.pad(X, [(0, 0)] * (X.ndim - 2) + [(0, 3), (0, 0)])
        return np.concatenate([T, X_padded], axis=-1)

    @staticmethod
    def transform(A: FloatingNDArray, T: FloatingNDArray) -> FloatingNDArray:
        # A is the expanded version of X_new, but for TPS expand(X, Y) returns the full matrix.
        # In find_transformation, A is passed to irls and ransac.
        # X_new is A[..., :-3, -2:]
        X_new = A[..., :-3, -2:]
        # T: [Optional Batch, N+3, 4] where last 2 cols are X_old
        X_old = T[..., :-3, -2:]
        W = T[..., :-3, :2]
        ab = T[..., -3:, :2]

        if T.ndim == 3:
            # RANSAC/Batch case
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
            return (K @ W) + (P_for_matmul @ ab)
        else:
            return (K @ W) + (P @ ab)
