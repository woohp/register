import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

import register
from register.transforms import (
    AffineTransform,
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


@pytest.mark.parametrize(
    "transform",
    [
        PerspectiveTransform(),
        AffineTransform(),
        RigidTransform(),
        SimilarityTransform(),
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
