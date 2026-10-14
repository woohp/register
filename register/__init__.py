from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .transforms import Transform
from .utils import hamming_distance_matrix

TARGET_MAX_SIZE = 1_200
MAX_KEYPOINTS = 2_000  # strongest BRISK responses kept per image
DESCRIPTOR_BITS = 512
# Hamming-distance margins (in bits) by which a match must beat the runner-up: an image keypoint's best template
# match must be clearly better than its second best, and a template keypoint's best image match must be unique.
IMAGE_MATCH_MARGIN = 10
TEMPLATE_MATCH_MARGIN = 1
# Match weights are exp(-MATCH_WEIGHT_DECAY * distance / DESCRIPTOR_BITS).
MATCH_WEIGHT_DECAY = 20

f32Array = NDArray[np.float32]
u8Array = NDArray[np.uint8]


def register_image(image1: u8Array, image2: u8Array, transform: Transform) -> Transform:
    keypoints1, descriptors1 = detect_features(image1)
    keypoints2, descriptors2 = detect_features(image2)
    _check_enough("keypoints in the image", len(keypoints1), transform)
    _check_enough("keypoints in the template", len(keypoints2), transform)

    X, Y, weights = do_match(keypoints1, descriptors1, keypoints2, descriptors2)
    _check_enough("matches", len(X), transform)

    transform.fit(Y, X, weights)
    return transform


def _check_enough(what: str, count: int, transform: Transform) -> None:
    if count < transform.min_samples:
        raise ValueError(f"Found {count} {what}, but {type(transform).__name__} needs at least {transform.min_samples}")


def resize_and_blur(image: u8Array) -> u8Array:
    image = aspect_aware_resize(image, TARGET_MAX_SIZE)
    # A 3x3 kernel truncates sigma=2 to weights of about (0.32, 0.36, 0.32), nearly a box blur: the kernel size, not
    # sigma, sets the blur strength. Kept as is; changing it would shift detection results.
    return cast(u8Array, cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=2.0, sigmaY=2.0))


def aspect_aware_resize(image: u8Array, max_size: float) -> u8Array:
    ratio = max_size / max(image.shape[:2])
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    return cast(u8Array, cv2.resize(image, (0, 0), None, ratio, ratio, interpolation))


def detect_features(image: u8Array) -> tuple[f32Array, u8Array]:
    """Detect keypoints on a downscaled copy of `image`; coordinates are normalized by its longest side."""
    max_size = max(image.shape[:2])
    small_image = resize_and_blur(image)

    detector = cv2.xfeatures2d.BRISK.create(octaves=5)
    # 5.0 is OpenCV's documented TEBLID sampling scale for BRISK keypoints.
    processor = cv2.xfeatures2d.TEBLID.create(5.0, cv2.xfeatures2d.TEBLID_SIZE_512_BITS)

    keypoints: Sequence[cv2.KeyPoint] = sorted(
        detector.detect(small_image, None), key=lambda kp: kp.response, reverse=True
    )
    keypoints = keypoints[:MAX_KEYPOINTS]
    keypoints, descriptors = processor.compute(small_image, keypoints)
    if not keypoints:  # descriptors is None rather than an empty array in this case
        return np.empty((0, 2), dtype=np.float32), np.empty((0, DESCRIPTOR_BITS // 8), dtype=np.uint8)
    keypoints_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    # cv2.resize aligns pixel centers, so a small-image coordinate maps back as (x + 0.5) / ratio - 0.5.
    ratio = TARGET_MAX_SIZE / max_size
    keypoints_array = ((keypoints_array + 0.5) / ratio - 0.5) / max_size
    return keypoints_array, cast(u8Array, descriptors)


def do_match(
    keypoints1: f32Array,
    descriptors1: u8Array,
    keypoints2: f32Array,
    descriptors2: u8Array,
) -> tuple[f32Array, f32Array, f32Array]:
    distances = hamming_distance_matrix(descriptors1.view(np.uint64), descriptors2.view(np.uint64))

    x_distances_indices = np.argpartition(distances, (1, 2), axis=1)[:, :2]
    y_distances_indices = np.argpartition(distances, (1, 2), axis=0)[:2, :]

    x_distances = np.take_along_axis(distances, x_distances_indices, 1)
    y_distances = np.take_along_axis(distances, y_distances_indices, 0)

    should_take_x = (x_distances[:, 1] - x_distances[:, 0]) >= IMAGE_MATCH_MARGIN
    should_take_y = (y_distances[1, :] - y_distances[0, :]) >= TEMPLATE_MATCH_MARGIN

    should_take_x[y_distances_indices[0, ~should_take_y]] = False

    y_indices = x_distances_indices[should_take_x, 0]
    weights = np.exp(distances[should_take_x, y_indices].astype(np.float32) * (-MATCH_WEIGHT_DECAY / DESCRIPTOR_BITS))

    X = keypoints1[should_take_x]
    Y = keypoints2[y_indices]

    weights_sort_indices = np.argsort(weights)[::-1]
    X = X[weights_sort_indices]
    Y = Y[weights_sort_indices]
    weights = weights[weights_sort_indices]

    X.flags.writeable = False
    Y.flags.writeable = False
    weights.flags.writeable = False

    return X, Y, weights
