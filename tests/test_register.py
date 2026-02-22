import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

import register


def test_register_against_self_should_produce_identity_transformation() -> None:
    img = np.asarray(Image.open("tests/assets/temp1.jpg"))
    T, loss = register.register_image(img, img, "affine")
    np.testing.assert_array_almost_equal(T, np.array([[0, 0], [1, 0], [0, 1]]), decimal=5)
    assert loss < 0.01


@pytest.mark.parametrize(
    "method",
    ["perspective", "affine", "rigid", "similarity", "thin_plate_spline", "quadratic_bernstein"],
)
def test_register_image1_transforms(images: tuple[NDArray[np.uint8], NDArray[np.uint8]], method: str) -> None:
    image1, template_image = images
    T, _loss = register.register_image(image1, template_image, method)  # type: ignore
    registered_image: NDArray[np.uint8] = register.generate_warped_image(image1, template_image.shape, T)
    Image.fromarray(registered_image).save(f"registered_{method}.png")
