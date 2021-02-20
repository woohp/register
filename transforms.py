from __future__ import annotations
import numpy as np
from typing import Literal

method_type = Literal['perspective', 'affine', 'quadratic']


class Transform:
    @staticmethod
    def expand(X: np.ndarray, Y: np.ndarray):
        pass

    @staticmethod
    def estimate(A: np.ndarray, Y: np.ndarray, lamb: float):
        pass

    @staticmethod
    def transform(A: np.ndarray, T: np.ndarray):
        pass

    @staticmethod
    def get_transform(method: method_type) -> type[Transform]:
        if method == 'perspective':
            return PerspectiveTransform
        elif method == 'affine':
            return AffineTransform
        elif method == 'quadratic':
            return QuadraticTransform


def poly_expand(X: np.ndarray, d: int) -> np.ndarray:
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
    def estimate(A: np.ndarray, Y: np.ndarray, lamb: float):
        # solve regularized least squares
        # https://stackoverflow.com/questions/27476933/numpy-linear-regression-with-regularization
        A_T = A.swapaxes(-2, -1)
        return np.linalg.solve(A_T @ A + lamb * np.identity(A.shape[-1], dtype=A.dtype), A_T @ Y)

    @staticmethod
    def transform(A: np.ndarray, T: np.ndarray):
        return A @ T


class AffineTransform(PolynomialTransform):
    @staticmethod
    def expand(X: np.ndarray, Y: np.ndarray):
        return poly_expand(X, 1)


class QuadraticTransform(PolynomialTransform):
    @staticmethod
    def expand(X: np.ndarray, Y: np.ndarray):
        return poly_expand(X, 2)


class PerspectiveTransform(Transform):
    @staticmethod
    def expand(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        ones = np.ones((N, 1), dtype=X.dtype)
        zeros = np.zeros((N, 3), dtype=X.dtype)
        row1 = np.concatenate((X, ones, zeros, -X * Y[:, :1], -Y[:, :1]), axis=-1)
        row2 = np.concatenate((zeros, X, ones, -X * Y[:, 1:], -Y[:, 1:]), axis=-1)
        return np.stack((row1, row2), axis=1)

    @staticmethod
    def estimate(A: np.ndarray, Y: np.ndarray, lamb: float) -> np.ndarray:
        _, _, V = np.linalg.svd(A.reshape(A.shape[0], -1, A.shape[-1]))
        Ts = V[:, -1].reshape(-1, 3, 3)
        Ts /= Ts[:, -1:, -1:]
        Ts = Ts.swapaxes(1, 2)
        return Ts

    @staticmethod
    def transform(A: np.ndarray, T: np.ndarray) -> np.ndarray:
        Y_hat = A[..., 0, :3] @ T
        Y_hat[Y_hat == 0] = np.finfo(Y_hat.dtype).eps
        Y_hat = Y_hat[..., :-1] / Y_hat[..., -1:]
        return Y_hat
