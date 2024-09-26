from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypeVar

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
        return np.linalg.solve(A_T @ A + lamb * np.identity(A.shape[-1], dtype=A.dtype), A_T @ Y)

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
    def _estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float, compute_scale: bool) -> FloatingNDArray:
        src_xy = A[..., 1:]
        dst_xy = Y

        # Compute centroids
        src_center: FloatingNDArray = np.mean(src_xy, axis=-2, keepdims=True)
        dst_center: FloatingNDArray = np.mean(dst_xy, axis=-2, keepdims=True)

        # Center the points
        src_centered = src_xy - src_center
        dst_centered = dst_xy - dst_center

        if compute_scale:
            src_scale: FloatingNDArray = np.sqrt(np.sum(src_centered**2, axis=(-2, -1), keepdims=True) / A.shape[-2])
            dst_scale: FloatingNDArray = np.sqrt(np.sum(dst_centered**2, axis=(-2, -1), keepdims=True) / Y.shape[-2])
            scale = dst_scale / src_scale

            src_normalized = src_centered / src_scale
            dst_normalized = dst_centered / dst_scale

        else:
            src_normalized = src_centered
            dst_normalized = dst_centered

        # Compute the covariance matrix
        H = src_normalized.swapaxes(-2, -1) @ dst_normalized

        # Compute the optimal rotation using SVD
        U, _, Vt = np.linalg.svd(H)
        R = Vt.swapaxes(-2, -1) @ U.swapaxes(-2, -1)

        # Ensure a proper rotation matrix (determinant = 1)
        needs_inverting = np.linalg.det(R) < 0
        Vt[needs_inverting, -1, :] *= -1
        R[needs_inverting] = Vt[needs_inverting].swapaxes(-2, -1) @ U[needs_inverting].swapaxes(-2, -1)

        if compute_scale:
            R *= scale

        # compute the translation
        t = dst_center[..., 0, :] - (R @ src_center.swapaxes(-2, -1))[..., 0]

        # Construct the full transformation matrix
        return np.concatenate([t[..., None, :], R.swapaxes(-2, -1)], axis=-2)

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
        pairwise_distance_squared = get_pairwise_distances_squared(X, Y)
        K = pairwise_distance_squared * np.log(pairwise_distance_squared.clip(min=1e-6))

        P = np.pad(X, [(0, 0)] * (X.ndim - 1) + [(1, 0)], constant_values=1)
        zeros = np.zeros((*X.shape[:-2], 3, 3), dtype=X.dtype)
        A = np.block([
            [K, P],
            [P.swapaxes(-1, -2), zeros],
        ])

        return A

    @staticmethod
    def select(
        A: FloatingNDArray, Y: FloatingNDArray, samples: NDArray[np.int64]
    ) -> tuple[FloatingNDArray, FloatingNDArray]:
        X = A[:-3, -2:]
        Y_sampled = Y[samples]
        A_sampled = ThinPlateSplineTransform.expand(X[samples], Y_sampled)
        return A_sampled, Y_sampled

    @staticmethod
    def estimate(A: FloatingNDArray, Y: FloatingNDArray, lamb: float) -> FloatingNDArray:
        Y_aug = np.pad(Y, [(0, 0)] * (Y.ndim - 2) + [(0, 3), (0, 0)], constant_values=0)
        T = np.linalg.solve(A, Y_aug)
        return np.concatenate([T, A[:, :, -2:]], axis=-1)

    @staticmethod
    def transform(A: FloatingNDArray, T: FloatingNDArray) -> FloatingNDArray:
        X_old = T[..., :-3, -2:]
        X_new = A[:-3, -2:]

        pairwise_distance_squared = get_pairwise_distances_squared(X_new, X_old)
        K = pairwise_distance_squared * np.log(pairwise_distance_squared.clip(min=1e-6))
        P = np.pad(X_new, [(0, 0)] * (X_new.ndim - 1) + [(1, 0)], constant_values=1)[None].repeat(T.shape[0], axis=0)

        T = T[..., :-2]
        return (K @ T[..., :-3, :]) + (P @ T[..., -3:, :])
