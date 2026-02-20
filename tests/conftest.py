import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def images():
    template_image = np.asarray(Image.open("tests/assets/temp1.jpg").convert("L"))
    image1 = np.asarray(Image.open("tests/assets/page1.jpg").convert("L"))
    return image1, template_image
