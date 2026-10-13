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
    ThinPlateSplineTransform,
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
        ThinPlateSplineTransform(),
        BernsteinTransform(degrees=2),
    ],
)
def test_register_image1_transforms(images: tuple[NDArray[np.uint8], NDArray[np.uint8]], transform: Transform) -> None:
    image1, template_image = images
    register.register_image(image1, template_image, transform)
    registered_image = transform.warp(image1, template_image.shape)
    method_name = type(transform).__name__.replace("Transform", "").lower()
    Image.fromarray(registered_image).save(f"registered_{method_name}.png")

    # Unregistered ink overlap is ~0.03; even rigid fits (the page has perspective) roughly double it.
    assert ink_dice(registered_image, template_image) > 1.5 * ink_dice(image1, template_image)


def ink_dice(image: NDArray[np.uint8], template: NDArray[np.uint8]) -> float:
    image_ink, template_ink = ink_mask(image), ink_mask(template)
    return 2 * (image_ink & template_ink).sum() / (image_ink.sum() + template_ink.sum())


def ink_mask(image: NDArray[np.uint8]) -> NDArray[np.bool]:
    return cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10).astype(bool)
