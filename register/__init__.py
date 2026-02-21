import random
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .transforms import QuadraticTransform, ThinPlateSplineTransform, Transform, method_type
from .utils import (
    generate_quadratic_remap_maps,
    generate_tps_remap_maps,
    get_distance_squared,
    hamming_distance_matrix,
)
from .utils.sampling import get_random_samples

TARGET_MAX_SIZE = 1_200
N_ITERS_RANSAC = 4_000
N_ITERS_IRLS = 50
SIGMA = 0.00003
RANSAC_L2_REGULARIZATION = 0.005

f32Array = NDArray[np.float32]
u8Array = NDArray[np.uint8]


def register_image(image1: u8Array, image2: u8Array, method: method_type = "perspective") -> tuple[f32Array, float]:
    image1 = resize_and_blur(image1)
    image2 = resize_and_blur(image2)

    keypoints1, descriptors1 = detect_features(image1)
    keypoints2, descriptors2 = detect_features(image2)

    X, Y, weights = do_match(keypoints1, descriptors1, keypoints2, descriptors2)
    transform_type = Transform.get_transform(method)

    # We always estimate the mapping from target (image2) to source (image1)
    # because image warping is a "pull" operation (iterating over target pixels).
    return find_transformation(Y, X, weights, transform_type)


def resize_and_blur(image: u8Array) -> u8Array:
    image = aspect_aware_resize(image, TARGET_MAX_SIZE)
    return cast(u8Array, cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=2.0, sigmaY=2.0))


def aspect_aware_resize(image: u8Array, max_size: float) -> u8Array:
    ratio = max_size / max(image.shape[:2])
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    return cast(u8Array, cv2.resize(image, (0, 0), None, ratio, ratio, interpolation))


def detect_features(image: u8Array) -> tuple[f32Array, u8Array]:
    detector = cv2.BRISK.create(octaves=5)
    processor = cv2.xfeatures2d.TEBLID.create(5.0, cv2.xfeatures2d.TEBLID_SIZE_512_BITS)

    keypoints: Sequence[cv2.KeyPoint] = sorted(detector.detect(image, None), key=lambda kp: kp.response, reverse=True)
    keypoints = keypoints[:2000]
    keypoints, descriptors = processor.compute(image, keypoints)
    keypoints_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    keypoints_array /= TARGET_MAX_SIZE
    return keypoints_array, cast(u8Array, descriptors)


def do_match(
    keypoints1: f32Array,
    descriptors1: u8Array,
    keypoints2: f32Array,
    descriptors2: u8Array,
) -> tuple[f32Array, f32Array, f32Array]:  # tuple of (X, Y, weights)
    # descriptors1: [M, 64]
    # descriptors2: [N, 64]
    distances = hamming_distance_matrix(descriptors1.view(np.uint64), descriptors2.view(np.uint64))  # [M, N]

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
    X: f32Array,
    Y: f32Array,
    weights: f32Array,
    transform: type[Transform],
) -> tuple[f32Array, float]:
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


def geman_mcclure_loss(d: f32Array, sigma: float) -> f32Array:
    return d / (d + sigma)


def ransac(A: f32Array, Y: f32Array, weights: f32Array, transform: type[Transform]) -> tuple[f32Array, float]:
    samples: NDArray[np.int64] = np.array(
        get_random_samples(Y.shape[0], transform.min_samples, N_ITERS_RANSAC, random.Random(4321)),
        dtype=np.int64,
    )

    b_A, b_Y = transform.select(A, Y, samples)

    Ts = transform.estimate(b_A, b_Y, RANSAC_L2_REGULARIZATION)

    # compute losses
    Y_hat = transform.transform(A, Ts)
    distance_squared = get_distance_squared(Y_hat, Y)
    objective_losses = np.average(geman_mcclure_loss(distance_squared, SIGMA), axis=-1, weights=weights)
    weights_losses = transform.penalty(Ts)
    losses = objective_losses + RANSAC_L2_REGULARIZATION * weights_losses

    best_idx = np.argmin(losses)
    return Ts[best_idx], losses[best_idx].item()


def irls(
    A: f32Array,
    Y: f32Array,
    weights: f32Array,
    T: f32Array,
    initial_loss: float,
    transform: type[Transform],
) -> tuple[f32Array, float]:
    """Iteratively reweighted least squares"""

    last_T = T
    last_loss = initial_loss

    Y_hat = transform.transform(A, T)
    w = weights / (get_distance_squared(Y_hat, Y) + SIGMA)

    for _ in range(N_ITERS_IRLS):
        # estimate a new transformation using the current weights
        w_ext = np.ones((A.shape[0],) + w.shape[1:], dtype=w.dtype)
        w_ext[: w.shape[0]] = w
        w_reshaped = w_ext.reshape([-1] + [1] * (A.ndim - 1))
        WA = A * w_reshaped
        WY = np.pad(Y, [(0, A.shape[0] - Y.shape[0]), (0, 0)], constant_values=0) * w_reshaped
        T = transform.estimate(WA, WY, RANSAC_L2_REGULARIZATION)

        # measure how good the new transformation is
        Y_hat = transform.transform(A, T)
        distance_squared = get_distance_squared(Y_hat, Y)
        w = weights / (distance_squared + SIGMA)
        objective_loss = (w * distance_squared).sum() / weights.sum()
        weights_loss = transform.penalty(T)
        loss: float = (objective_loss + weights_loss * RANSAC_L2_REGULARIZATION).item()

        # if the new transformation is not better, return the old transformatoin
        if loss >= last_loss:
            return last_T, last_loss
        elif last_loss - loss < 1e-6:
            return T, loss

        last_T = T
        last_loss = loss

    return T, last_loss


def generate_warped_image(image: u8Array, template_shape: tuple[int, int], T: f32Array) -> u8Array:
    image_max_size: int = max(image.shape[:2])
    template_height, template_width = template_shape
    out_size = (template_width, template_height)
    template_max_size: int = max(template_shape)

    # All transformations T are assumed to map from Target (normalized) to Source (normalized)
    T = T.copy()

    if T.shape == (3, 2):
        # Affine mapping: Source = [1, x_t, y_t] @ T
        # Scale T to map from Target pixels to Source pixels
        T[0] *= image_max_size
        T[1:] *= image_max_size / template_max_size

        # OpenCV warpAffine expects a 2x3 matrix: [ [a, b, tx], [c, d, ty] ]
        # T is [ [tx, ty], [a, c], [b, d] ] -> we need to roll and transpose
        M = np.roll(T, -1, axis=0).T
        out = cv2.warpAffine(image, M, dsize=out_size, flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)

    elif T.shape == (6, 2):
        # Quadratic mapping: use cv2.remap
        T[0] *= image_max_size
        T[1:3] *= image_max_size / template_max_size
        T[3:] *= image_max_size / (template_max_size**2)

        # Roll to put translation last if needed? No, generate_quadratic_remap_maps expects this order:
        # T[0] is const, T[1] is x, T[2] is y, T[3] is xx, T[4] is xy, T[5] is yy
        map1, map2 = generate_quadratic_remap_maps(*template_shape, T)
        out = cv2.remap(image, map1, map2, interpolation=cv2.INTER_AREA)

    elif T.shape == (3, 3):
        # Perspective mapping
        # T maps normalized Target to normalized Source.
        # ScaledPerspective = S_source @ T @ S_target_inv
        # where S(x) = x / image_max_size
        T[:2, :2] *= image_max_size / template_max_size
        T[2, :2] *= image_max_size
        T[:2, 2] /= template_max_size
        # T[2, 2] remains unchanged (usually 1)

        # OpenCV warpPerspective expects M s.t. x_s = M @ x_t. We have T s.t. x_s = x_t @ T
        out = cv2.warpPerspective(image, T.T, dsize=out_size, flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)

    elif T.ndim == 2 and T.shape[1] == 4:
        # Thin Plate Spline
        map1, map2 = generate_tps_remap_maps(*template_shape, T)
        map1 *= image_max_size
        map2 *= image_max_size
        out = cv2.remap(image, map1, map2, interpolation=cv2.INTER_AREA)

    else:
        raise ValueError(f"unrecognized transformation: {T!r}")

    return cast(u8Array, out)
