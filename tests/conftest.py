import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image


@pytest.fixture
def images() -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    template = np.asarray(Image.open("tests/assets/temp1.jpg").convert("L"))
    image = np.asarray(Image.open("tests/assets/page1.jpg").convert("L"))
    return image, template
