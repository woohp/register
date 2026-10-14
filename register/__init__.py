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


def register_image[T: Transform](image: u8Array, template: u8Array, transform: T) -> T:
    """Fit `transform` to map template coordinates to image coordinates and return it.

    Read `transform.loss_` to reject bad registrations, and `transform.warp` to align the image to the template.
    """
    image_keypoints, image_descriptors = detect_features(image)
    template_keypoints, template_descriptors = detect_features(template)
    _check_enough("keypoints in the image", len(image_keypoints), transform)
    _check_enough("keypoints in the template", len(template_keypoints), transform)

    image_points, template_points, weights = do_match(
        image_keypoints, image_descriptors, template_keypoints, template_descriptors
    )
    _check_enough("matches", len(image_points), transform)

    return transform.fit(template_points, image_points, weights)


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
    image_keypoints: f32Array,
    image_descriptors: u8Array,
    template_keypoints: f32Array,
    template_descriptors: u8Array,
) -> tuple[f32Array, f32Array, f32Array]:
    # Rows are image keypoints, columns are template keypoints.
    distances = hamming_distance_matrix(image_descriptors.view(np.uint64), template_descriptors.view(np.uint64))

    image_top2 = np.argpartition(distances, (1, 2), axis=1)[:, :2]
    template_top2 = np.argpartition(distances, (1, 2), axis=0)[:2, :]

    image_top2_distances = np.take_along_axis(distances, image_top2, 1)
    template_top2_distances = np.take_along_axis(distances, template_top2, 0)

    image_unambiguous = (image_top2_distances[:, 1] - image_top2_distances[:, 0]) >= IMAGE_MATCH_MARGIN
    template_unambiguous = (template_top2_distances[1, :] - template_top2_distances[0, :]) >= TEMPLATE_MATCH_MARGIN

    # Keep mutual nearest neighbors only, so every template keypoint is matched at most once.
    best_template = image_top2[:, 0]
    mutual = template_top2[0, best_template] == np.arange(len(distances))
    keep_image = image_unambiguous & mutual & template_unambiguous[best_template]

    matched_template = best_template[keep_image]
    weights = np.exp(
        distances[keep_image, matched_template].astype(np.float32) * (-MATCH_WEIGHT_DECAY / DESCRIPTOR_BITS)
    )

    image_points = image_keypoints[keep_image]
    template_points = template_keypoints[matched_template]

    order = np.argsort(weights)[::-1]
    image_points = image_points[order]
    template_points = template_points[order]
    weights = weights[order]

    image_points.flags.writeable = False
    template_points.flags.writeable = False
    weights.flags.writeable = False

    return image_points, template_points, weights
