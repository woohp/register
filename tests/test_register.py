import numpy as np
import pytest
from PIL import Image

import register


def test_register_against_self_should_produce_identity_transformation() -> None:
    img = np.asarray(Image.open("tests/assets/temp1.jpg"))
    T, loss = register.register_image(img, img, "affine")
    np.testing.assert_array_almost_equal(T, np.array([[0, 0], [1, 0], [0, 1]]), decimal=5)
    assert loss < 0.01


@pytest.mark.parametrize(
    "method",
    [
        "perspective",
        "affine",
        "rigid",
        "similarity",
        "quadratic",
        pytest.param("thin_plate_spline", marks=pytest.mark.xfail(reason="currently broken")),
    ],
)
def test_register_image1_transforms(images, method) -> None:
    image1, template_image = images
    T, _loss = register.register_image(image1, template_image, method)
    registered_image = register.generate_warped_image(image1, template_image.shape, T)  # type: ignore
    Image.fromarray(registered_image).save(f"registered_{method}.png")
