from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Literal

method_type = Literal['perspective', 'affine', 'quadratic']


class Transform:
    @staticmethod
    def expand(X: ndarray, Y: ndarray):
        pass

    @staticmethod
    def estimate(A: ndarray, Y: ndarray, lamb: float):
        pass

    @staticmethod
    def transform(A: ndarray, T: ndarray):
        pass

    @staticmethod
    def get_transform(method: method_type) -> type[Transform]:
        if method == 'perspective':
            return PerspectiveTransform
        elif method == 'affine':
            return AffineTransform
        elif method == 'quadratic':
            return QuadraticTransform


def poly_expand(X: ndarray, d: int) -> ndarray:
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
    def estimate(A: ndarray, Y: ndarray, lamb: float):
        # A: [..., samples, K] where K is determined by the degree
        # Y: [..., samples, 2]
        # solve regularized least squares
        # https://stackoverflow.com/questions/27476933/numpy-linear-regression-with-regularization
        A_T = A.swapaxes(-2, -1)
        return np.linalg.solve(A_T @ A + lamb * np.identity(A.shape[-1], dtype=A.dtype), A_T @ Y)

    @staticmethod
    def transform(A: ndarray, T: ndarray):
        return A @ T


class AffineTransform(PolynomialTransform):
    @staticmethod
    def expand(X: ndarray, Y: ndarray):
        return poly_expand(X, 1)


class QuadraticTransform(PolynomialTransform):
    @staticmethod
    def expand(X: ndarray, Y: ndarray):
        return poly_expand(X, 2)


class PerspectiveTransform(Transform):
    @staticmethod
    def expand(X: ndarray, Y: ndarray) -> ndarray:
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
    def estimate(A: ndarray, Y: ndarray, lamb: float) -> ndarray:
        # A: [..., 2, 9]
        # Y: [..., 2], not used
        _, _, V = np.linalg.svd(A.reshape(A.shape[:-3] + (-1, 9)))
        Ts = V[:, -1].reshape(-1, 3, 3)
        Ts /= Ts[:, -1:, -1:]
        Ts = Ts.swapaxes(1, 2)
        return Ts

    @staticmethod
    def transform(A: ndarray, T: ndarray) -> ndarray:
        Y_hat = A[..., 0, :3] @ T
        Y_hat[Y_hat == 0] = np.finfo(Y_hat.dtype).eps
        Y_hat = Y_hat[..., :-1] / Y_hat[..., -1:]
        return Y_hat
