import random

import cv2
import numpy as np
from numpy import ndarray

from ._ext import generate_quadratic_remap_maps, hamming_distance_matrix
from .transforms import QuadraticTransform, Transform, method_type
from .utils.sampling import get_random_samples

TARGET_MAX_SIZE = 1_200
N_ITERS_RANSAC = 4_000
N_ITERS_IRLS = 50
SIGMA = 0.00003
RANSAC_L2_REGULARIZATION = 0.005


def register_image(image1: ndarray, image2: ndarray, method: method_type = 'perspective'):
    image1 = resize_and_blur(image1)
    image2 = resize_and_blur(image2)

    keypoints1, descriptors1 = detect_features(image1)
    keypoints2, descriptors2 = detect_features(image2)

    X, Y, weights = do_match(keypoints1, descriptors1, keypoints2, descriptors2)
    return find_transformation(X, Y, weights, Transform.get_transform(method))


def resize_and_blur(image: ndarray) -> ndarray:
    image = aspect_aware_resize(image, TARGET_MAX_SIZE)
    image = cv2.GaussianBlur(image, (3, 3), 2, 2)
    return image


def aspect_aware_resize(image: ndarray, max_size: float) -> ndarray:
    ratio = max_size / max(image.shape[:2])
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (0, 0), None, ratio, ratio, interpolation)


def detect_features(image: ndarray) -> tuple[ndarray, ndarray]:
    detector = cv2.BRISK.create(None, 5)
    processor = cv2.xfeatures2d.BEBLID_create(5.0, cv2.xfeatures2d.BEBLID_SIZE_512_BITS)

    keypoints = sorted(detector.detect(image, None), key=lambda kp: kp.response, reverse=True)
    keypoints = keypoints[:2000]
    keypoints, descriptors = processor.compute(image, keypoints)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32) / TARGET_MAX_SIZE, descriptors


def do_match(
    keypoints1: ndarray,
    descriptors1: ndarray,
    keypoints2: ndarray,
    descriptors2: ndarray,
) -> tuple[ndarray, ndarray, ndarray]:
    # descriptors1: [M, 64]
    # descriptors2: [N, 64]
    distances = hamming_distance_matrix(descriptors1, descriptors2)  # [M, N]

    x_distances_indices = np.argpartition(distances, (1, 2), axis=1)[:, :2]  # [M, 2]
    y_distances_indices = np.argpartition(distances, (1, 2), axis=0)[:2, :]  # [2, N]

    x_distances = np.take_along_axis(distances, x_distances_indices, 1)  # [M, 2]
    y_distances = np.take_along_axis(distances, y_distances_indices, 0)  # [2, N]

    # enforce ratio constraint, where the best match must be at least some amount better than the 2nd best
    should_take_x = (x_distances[:, 1] - x_distances[:, 0]) >= 10
    should_take_y = (y_distances[1, :] - y_distances[0, :]) >= 1  # not a typo

    # enforce symmetry constraint
    should_take_x[y_distances_indices[0, ~should_take_y]] = False

    y_indices = x_distances_indices[should_take_x, 0]
    weights = np.exp(distances[should_take_x, y_indices].astype(np.float32) * (-20 / 512))

    X = keypoints1[should_take_x]
    Y = keypoints2[y_indices]

    # sort by weights in descending order
    weights_sort_indices = np.argsort(weights)[::-1]
    X = X[weights_sort_indices]
    Y = Y[weights_sort_indices]
    weights = weights[weights_sort_indices]

    # make them immutable
    X.flags.writeable = False
    Y.flags.writeable = False
    weights.flags.writeable = False

    return X, Y, weights


def find_transformation(
    X: ndarray,
    Y: ndarray,
    weights: ndarray,
    transform: type[Transform],
) -> tuple[ndarray, float]:
    A = transform.expand(X, Y)

    # if the transform is quadratic, then expand transformation to quadratic, and repeat the IRLS step
    if transform is QuadraticTransform:
        T, loss = ransac(A[:, :3], Y, weights, transform)
        T, loss = irls(A[:, :3], Y, weights, T, loss, transform)
        T = np.pad(T, [(0, 3), (0, 0)])
        T, loss = irls(A, Y, weights, T, loss, transform)

    else:
        T, loss = ransac(A, Y, weights, transform)
        T, loss = irls(A, Y, weights, T, loss, transform)

    T.flags.writeable = False

    return T, loss


def geman_mcclure_loss(d: ndarray, sigma: float) -> ndarray:
    return d / (d + sigma)


def ransac(A: ndarray, Y: ndarray, weights: ndarray, transform: type[Transform]) -> tuple[ndarray, float]:
    samples = get_random_samples(A.shape[0], 4, N_ITERS_RANSAC, random.Random(4321))

    b_A = A[samples]  # [batch-size, ...]
    b_Y = Y[samples]  # [batch-size, samples, 2]

    Ts = transform.estimate(b_A, b_Y, RANSAC_L2_REGULARIZATION)

    # compute losses
    Y_hat = transform.transform(A, Ts)
    distances = ((Y_hat - Y)**2).sum(axis=-1)
    objective_losses = np.average(geman_mcclure_loss(distances, SIGMA), axis=-1, weights=weights)
    weights_losses = (Ts[:, 1:]**2).sum(axis=(1, 2))
    losses = objective_losses + RANSAC_L2_REGULARIZATION * weights_losses

    best_idx = np.argmin(losses)
    return Ts[best_idx], losses[best_idx].item()


def irls(
    A: ndarray,
    Y: ndarray,
    weights: ndarray,
    T: ndarray,
    initial_loss: float,
    transform: type[Transform],
) -> tuple[ndarray, float]:
    last_T = T
    last_loss = initial_loss

    Y_hat = transform.transform(A, T)
    distances = ((Y_hat - Y)**2).sum(axis=-1)
    w = weights / (distances + SIGMA)

    for iter_idx in range(N_ITERS_IRLS):
        # estimate a new transformation using the current weights
        w = w.reshape([-1] + [1] * (A.ndim - 1))
        WA = A * w
        WY = Y * w
        T = transform.estimate(WA, WY, 0.0)

        # measure how good the new transformation is
        Y_hat = transform.transform(A, T)
        distances = ((Y_hat - Y)**2).sum(axis=-1)
        w = weights / (distances + SIGMA)
        objective_loss = (w * distances).sum() / weights.sum()
        weights_loss = (T[:, 1:]**2).sum(axis=(-2, -1))
        loss: float = (objective_loss + weights_loss * RANSAC_L2_REGULARIZATION).item()

        # if the new transformation is not better, return the old transformatoin
        if loss >= last_loss:
            return last_T, last_loss
        elif last_loss - loss < 1e-6:
            return T, loss

        last_T = T
        last_loss = loss

    return T, last_loss


def generate_warped_image(image: ndarray, template_shape: tuple[int, int], T):
    image_max_size: int = max(image.shape[:2])
    template_height, template_width = template_shape
    out_size = (template_width, template_height)
    template_max_size: int = max(template_shape)

    if T.shape == (3, 2):
        # If this is an affine transformation, we can simply call cv2.warpAffine
        # However, we must massage the matrix a bit
        T = T.copy()
        T[:1] *= template_max_size
        T[1:] *= template_max_size / image_max_size
        T = np.roll(T, -1, axis=0).T

        return cv2.warpAffine(image, T, out_size, cv2.INTER_AREA)

    elif T.shape == (6, 2):
        # If this is a quadratic transformation, use cv2.remap
        T = np.roll(T, -1, axis=0)
        T_expanded = np.identity(6, dtype=T.dtype)
        T_expanded[:6, :2] = T
        T = np.roll(np.linalg.inv(T_expanded)[:, :2], 1, axis=0)
        T *= image_max_size
        T[1:] /= template_max_size
        T[3:] /= template_max_size

        dest_grid = generate_quadratic_remap_maps(*template_shape, T)

        return cv2.remap(image, *dest_grid, cv2.INTER_AREA)

    elif T.shape == (3, 3):
        # massage the transformation matrix a bit
        T = T.copy()
        T[:2] *= image_max_size / template_max_size
        T[2] *= image_max_size
        T[:, -1] /= template_max_size

        return cv2.warpPerspective(image, T.T, out_size, cv2.INTER_AREA)
