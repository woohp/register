import cv2
import numpy as np
from utils.sampling import get_random_samples
import _ext
from typing import List, Tuple


TARGET_MAX_SIZE = 1_200
N_ITERS_RANSAC = 4_000
N_ITERS_IRLS = 50
SIGMA = 0.00003
RANSAC_L2_REGULARIZATION = 0.005


def register_image(image1: np.ndarray, image2: np.ndarray):
    image1 = resize_and_blur(image1)
    image2 = resize_and_blur(image2)

    keypoints1, descriptors1 = detect_features(image1)
    keypoints2, descriptors2 = detect_features(image2)

    X, Y, weights = do_match(keypoints1, descriptors1, keypoints2, descriptors2)
    return find_transformation(X, Y, weights)


def resize_and_blur(image: np.ndarray) -> np.ndarray:
    image = aspect_aware_resize(image, TARGET_MAX_SIZE)
    image = cv2.GaussianBlur(image, (3, 3), 2, 2)
    return image


def aspect_aware_resize(image: np.ndarray, max_size: float) -> np.ndarray:
    ratio = max_size / max(image.shape[:2])
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (0, 0), None, ratio, ratio, interpolation)


def detect_features(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    brisk = cv2.BRISK.create(None, 5)
    keypoints = brisk.detect(image)
    keypoints.sort(key=lambda kp: kp.response, reverse=True)
    keypoints = keypoints[:2000]
    keypoints, descriptors = brisk.compute(image, keypoints)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32) / TARGET_MAX_SIZE, descriptors


def do_match(
    keypoints1: np.ndarray,
    descriptors1: np.ndarray,
    keypoints2: np.ndarray,
    descriptors2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = _ext.hamming_distance_matrix(descriptors1, descriptors2)
    breakpoint()

    x_distances_indices = np.argpartition(distances, (1, 2), axis=1)[:, :2]
    y_distances_indices = np.argpartition(distances, (1, 2), axis=0)[:2, :]

    x_distances = np.take_along_axis(distances, x_distances_indices, 1)
    y_distances = np.take_along_axis(distances, y_distances_indices, 0)

    should_take_x = (x_distances[:, 1] - x_distances[:, 0]) >= 10
    should_take_y = (y_distances[1, :] - y_distances[0, :]) >= 1  # not a typo

    # enforce symmetry constraint
    should_take_x[y_distances_indices[0, np.where(~should_take_y)[0]]] = False

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


def find_transformation(X: np.ndarray, Y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float]:
    A = poly_expand(X, 2)

    breakpoint()
    T, loss = ransac(A[:, :3], Y, weights)
    T, loss = irls(A[:, :3], Y, weights, T, loss)

    # expand transformation to quadratic, and repeat the IRLS step
    breakpoint()
    T = np.pad(T, [(0, 3), (0, 0)])
    T, loss = irls(A, Y, weights, T, loss)

    return T, loss


def poly_expand(X: np.ndarray, d: int):
    o = np.ones(X.shape[0], dtype=X.dtype)

    if d == 1:
        return np.column_stack((o, X))
    elif d == 2:
        x = X[:, 0]
        y = X[:, 1]
        xx = np.square(x)
        yy = np.square(y)
        xy = x * y
        return np.column_stack((o, X, xx, xy, yy))
    else:
        raise NotImplementedError


def geman_mcclure_loss(d: np.ndarray, sigma: float) -> np.ndarray:
    return d / (d + sigma)


def ransac(A: np.ndarray, Y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float]:
    samples: List[Tuple[int, int, int]] = []
    for i in range(N_ITERS_RANSAC):
        sample = get_random_samples(i, A.shape[0])
        if sample is None:
            break
        samples.append(sample)

    b_A = A[samples]  # [batch-size, 3, 3]
    b_Y = Y[samples]  # [batch-size, 3, 2]

    # solve regularized least squares
    # https://stackoverflow.com/questions/27476933/numpy-linear-regression-with-regularization
    b_A_T = b_A.swapaxes(1, 2)
    Ts = np.linalg.solve(b_A_T @ b_A + RANSAC_L2_REGULARIZATION * np.identity(3, dtype=b_A.dtype), b_A_T @ b_Y)

    # compute losses
    distances = ((A @ Ts - Y)**2).sum(axis=-1)
    objective_losses = (geman_mcclure_loss(distances, SIGMA) * weights).sum(axis=-1) / weights.sum()
    weights_losses = (Ts[:, 1:]**2).sum(axis=(1, 2))
    losses = objective_losses + RANSAC_L2_REGULARIZATION * weights_losses

    best_idx = np.argmax(losses)
    return Ts[best_idx], losses[best_idx].item()


def irls(A: np.ndarray, Y: np.ndarray, weights: np.ndarray, T: np.ndarray,
         initial_loss: float) -> Tuple[np.ndarray, float]:
    last_T = T
    last_objective_loss = initial_loss
    last_full_loss = initial_loss

    for iter_idx in range(N_ITERS_IRLS):
        distances = ((A @ T - Y)**2).sum(axis=-1)
        w = weights / (distances + SIGMA)
        WA = A * w[:, np.newaxis]
        WY = Y * w[:, np.newaxis]
        objective_loss = (w * distances).sum() / weights.sum()

        full_loss = objective_loss

        if iter_idx > 0:
            if full_loss > last_full_loss:
                T = last_T
                break
            elif last_full_loss - full_loss < 1e-6:
                last_objective_loss = objective_loss.item()
                break

        last_T = T
        last_objective_loss = objective_loss
        last_full_loss = full_loss.item()

        if iter_idx == N_ITERS_IRLS - 1:
            break

        T = np.linalg.lstsq(WA, WY, rcond=None)[0]

    return T, last_objective_loss


def generate_warped_image(image: np.ndarray, template_shape: Tuple[int, int], T):
    image_max_size: int = max(image.shape[:2])
    template_height, template_width = template_shape
    template_max_size: int = max(template_shape)

    if T.shape == (3, 2):
        # If this is an affine transformation, we can simply call cv2.warpAffine
        # However, we must massage the matrix a bit
        T = T.copy()
        T[:1] *= template_max_size
        T[1:] *= template_max_size / image_max_size
        T = np.roll(T, -1, axis=0).T

        return cv2.warpAffine(image, T, tuple(reversed(template_shape)), cv2.INTER_AREA, cv2.BORDER_CONSTANT, 0)

    elif T.shape == (6, 2):
        T = np.roll(T, -1, axis=0)
        T_expanded = np.identity(6, dtype=T.dtype)
        T_expanded[:6, :2] = T
        T = np.roll(np.linalg.inv(T_expanded)[:, :2], 1, axis=0)

        grid = np.mgrid[:template_height, :template_width].astype(np.float32) / template_max_size
        grid = grid.transpose(2, 1, 0)
        grid = grid.reshape(-1, 2)
        grid = poly_expand(grid, 2)

        dest_grid = (grid @ T).reshape(template_width, template_height, 2)
        dest_grid *= image_max_size

        return cv2.remap(image, dest_grid[..., 0], dest_grid[..., 1], cv2.INTER_AREA)
