import unittest

import numpy as np
from PIL import Image

import register


class RegisterTestCase(unittest.TestCase):
    def test_register_against_self_should_produce_identity_transformation(self) -> None:
        img = np.asarray(Image.open("tests/assets/temp1.jpg"))
        T, loss = register.register_image(img, img, "affine")
        np.testing.assert_array_almost_equal(T, np.array([[0, 0], [1, 0], [0, 1]]))
        self.assertLess(loss, 0.01)

    def test_register_image1_perspective_transform(self) -> None:
        template_image = np.asarray(Image.open("tests/assets/temp1.jpg").convert("L"))
        image1 = np.asarray(Image.open("tests/assets/page1.jpg").convert("L"))
        T, loss = register.register_image(image1, template_image, "perspective")
        registered_image = register.generate_warped_image(image1, template_image.shape, T)  # type: ignore
        Image.fromarray(registered_image).save("registered.png")

    def test_register_image1_affine_transform(self) -> None:
        template_image = np.asarray(Image.open("tests/assets/temp1.jpg").convert("L"))
        image1 = np.asarray(Image.open("tests/assets/page1.jpg").convert("L"))
        T, loss = register.register_image(image1, template_image, "affine")
        registered_image = register.generate_warped_image(image1, template_image.shape, T)  # type: ignore
        Image.fromarray(registered_image).save("registered.png")

    def test_register_image1_quadratic_transform(self) -> None:
        template_image = np.asarray(Image.open("tests/assets/temp1.jpg").convert("L"))
        image1 = np.asarray(Image.open("tests/assets/page1.jpg").convert("L"))
        T, loss = register.register_image(image1, template_image, "quadratic")
        registered_image = register.generate_warped_image(image1, template_image.shape, T)  # type: ignore
        Image.fromarray(registered_image).save("registered.png")
