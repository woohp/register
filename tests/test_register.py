from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

import register
from register.transforms import (
    AffineTransform,
    AnisotropicSimilarityTransform,
    BernsteinTransform,
    PerspectiveTransform,
    RigidTransform,
    SimilarityTransform,
    Transform,
)


def test_register_against_self_should_produce_identity_transformation() -> None:
    img = np.asarray(Image.open("tests/assets/temp1.jpg"))
    transform = register.register_image(img, img, AffineTransform())
    np.testing.assert_array_almost_equal(transform.matrix, np.array([[0, 0], [1, 0], [0, 1]]), decimal=5)


def test_register_featureless_image_raises() -> None:
    template = np.asarray(Image.open("tests/assets/temp1.jpg").convert("L"))
    blank = np.full_like(template, 128)
    with pytest.raises(ValueError, match="Found 0 keypoints in the image"):
        register.register_image(blank, template, AffineTransform())


@pytest.mark.parametrize(
    "transform",
    [
        PerspectiveTransform(),
        AffineTransform(),
        RigidTransform(),
        SimilarityTransform(),
        AnisotropicSimilarityTransform(),
        BernsteinTransform(degrees=2),
    ],
)
def test_register_image1_transforms(images: tuple[NDArray[np.uint8], NDArray[np.uint8]], transform: Transform) -> None:
    image, template = images
    register.register_image(image, template, transform)
    registered_image = transform.warp(image, template.shape)
    method_name = type(transform).__name__.replace("Transform", "").lower()
    Image.fromarray(registered_image).save(f"registered_{method_name}.png")

    # Unregistered ink overlap is ~0.03; even rigid fits (the page has perspective) roughly double it.
    assert ink_dice(registered_image, template) > 1.5 * ink_dice(image, template)


def ink_dice(image: NDArray[np.uint8], template: NDArray[np.uint8]) -> float:
    image_ink, template_ink = ink_mask(image), ink_mask(template)
    return 2 * (image_ink & template_ink).sum() / (image_ink.sum() + template_ink.sum())


def ink_mask(image: NDArray[np.uint8]) -> NDArray[np.bool]:
    return cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10).astype(bool)


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_register_rescaled_copy_maps_pixel_centers(scale: float) -> None:
    template = np.asarray(Image.open("tests/assets/temp1.jpg").convert("L"))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    image = cast(NDArray[np.uint8], cv2.resize(template, None, fx=scale, fy=scale, interpolation=interpolation))
    transform = register.register_image(image, template, AffineTransform())

    ys, xs = np.mgrid[0 : template.shape[0] : 256, 0 : template.shape[1] : 256]
    template_points = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    T = transform.matrix
    image_points = (T[0] + template_points / max(template.shape) @ T[1:]) * max(image.shape)

    # cv2.resize aligns pixel centers: template pixel x lands on image pixel (x + 0.5) * scale - 0.5.
    np.testing.assert_allclose(image_points, (template_points + 0.5) * scale - 0.5, atol=0.15)


def test_do_match_keeps_only_mutual_nearest_neighbors() -> None:
    rng = np.random.default_rng(0)
    template_descriptors = rng.integers(0, 256, (3, register.DESCRIPTOR_BITS // 8), dtype=np.uint8)
    near_copy = template_descriptors[0].copy()
    near_copy[:5] ^= 1  # 5 bits away from template descriptor 0
    # Image keypoints 0 and 1 both have template keypoint 0 as their best match; only image keypoint 0 is mutual.
    image_descriptors = np.stack([template_descriptors[0], near_copy, template_descriptors[1], template_descriptors[2]])
    image_keypoints = np.arange(8, dtype=np.float32).reshape(4, 2)
    template_keypoints = np.arange(6, dtype=np.float32).reshape(3, 2) + 100

    image_points, template_points, _ = register.do_match(
        image_keypoints, image_descriptors, template_keypoints, template_descriptors
    )

    matches = {(tuple(i), tuple(t)) for i, t in zip(image_points.tolist(), template_points.tolist())}
    expected = {
        (tuple(image_keypoints[i].tolist()), tuple(template_keypoints[t].tolist())) for i, t in [(0, 0), (2, 1), (3, 2)]
    }
    assert matches == expected
