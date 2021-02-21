import cv2
import numpy as np
from numpy import ndarray
import random
from transforms import Transform, method_type
from utils.sampling import get_random_samples
import _ext

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
    brisk = cv2.BRISK.create(None, 5)
    keypoints = brisk.detect(image)
    keypoints.sort(key=lambda kp: kp.response, reverse=True)
    keypoints = keypoints[:2000]
    keypoints, descriptors = brisk.compute(image, keypoints)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32) / TARGET_MAX_SIZE, descriptors


def do_match(
    keypoints1: ndarray,
    descriptors1: ndarray,
    keypoints2: ndarray,
    descriptors2: ndarray,
) -> tuple[ndarray, ndarray, ndarray]:
    # descriptors1: [M, 64]
    # descriptors2: [N, 64]
    distances = _ext.hamming_distance_matrix(descriptors1, descriptors2)  # [M, N]

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
    T, loss = ransac(A, Y, weights, transform)

    # expand transformation to quadratic, and repeat the IRLS step
    # T = np.pad(T, [(0, 3), (0, 0)])
    # T, loss = irls(A, Y, weights, T, loss)

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


def irls(A: ndarray, Y: ndarray, weights: ndarray, T: ndarray, initial_loss: float) -> tuple[ndarray, float]:
    last_T = T
    last_objective_loss = initial_loss
    last_full_loss = initial_loss

    for iter_idx in range(N_ITERS_IRLS):
        Y_hat = A[:, 0, :3] @ T
        Y_hat[Y_hat == 0] = np.finfo(Y_hat.dtype).eps
        Y_hat = Y_hat[..., :-1] / Y_hat[..., -1:]
        distances = ((Y_hat - Y)**2).sum(axis=-1)
        w = weights / (distances + SIGMA)
        WA = A * w[:, np.newaxis, np.newaxis]
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

        _, _, V = np.linalg.svd(WA.reshape(-1, 9))
        T = V[-1].reshape(3, 3)
        T /= T[-1:, -1:]
        T = T.swapaxes(0, 1)
        # T = np.linalg.lstsq(WA, WY, rcond=None)[0]

    return T, last_objective_loss


def generate_warped_image(image: ndarray, template_shape: tuple[int, int], T):
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

        return cv2.warpAffine(image, T, tuple(reversed(template_shape)), cv2.INTER_AREA)

    elif T.shape == (6, 2):
        # If this is a quadratic transformation, use cv2.remap
        T = np.roll(T, -1, axis=0)
        T_expanded = np.identity(6, dtype=T.dtype)
        T_expanded[:6, :2] = T
        T = np.roll(np.linalg.inv(T_expanded)[:, :2], 1, axis=0)
        T *= image_max_size
        T[1:] /= template_max_size
        T[3:] /= template_max_size

        dest_grid = _ext.generate_quadratic_remap_maps(*template_shape, T)

        return cv2.remap(image, *dest_grid, cv2.INTER_AREA)

    elif T.shape == (3, 3):
        # massage the transformation matrix a bit
        T = T.copy()
        T[:2] *= image_max_size / template_max_size
        T[2] *= image_max_size
        T[:, -1] /= template_max_size

        return cv2.warpPerspective(image, T.T, tuple(reversed(template_shape)), cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
