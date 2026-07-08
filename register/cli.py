from typing import Annotated, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from typer import Option, Typer
from skimage.filters import threshold_local

from register import generate_warped_image, register_image
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

u8Array = NDArray[np.uint8]

TRANSFORMS: dict[str, Transform] = {
    "perspective": PerspectiveTransform(),
    "affine": AffineTransform(),
    "rigid": RigidTransform(),
    "similarity": SimilarityTransform(),
    "anisotropic_similarity": AnisotropicSimilarityTransform(),
    "thin_plate_spline": ThinPlateSplineTransform(),
}

app = Typer()


@app.command()
def main(
    image_path: str,
    template_path: str,
    method: Annotated[str, Option("--method", "-m")] = "perspective",
    degrees: Annotated[int | None, Option("--degrees", "-d")] = 2,
    out_path: Annotated[str, Option("--out", "-o")] = "registered.png",
) -> None:
    if method == "bernstein":
        if degrees is None:
            raise ValueError("--degrees is required for bernstein method")
        transform: Transform = BernsteinTransform(degrees=degrees)
    elif method in TRANSFORMS:
        transform = TRANSFORMS[method]
    else:
        raise ValueError(f"Unknown method: {method!r}. Available: {list(TRANSFORMS) + ['bernstein']}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    if template is None:
        raise ValueError(f"Could not read template from {template_path}")

    register_image(cast(u8Array, image), cast(u8Array, template), transform)
    print("matrix:")
    print(transform.matrix)

    registered_image = generate_warped_image(cast(u8Array, image), template.shape, transform)
    cv2.imwrite(out_path, registered_image)

    blended = color_blend_3c(registered_image, template)
    Image.fromarray(blended).save("blended.png")


def color_blend_3c(img1, img2, block_size=35, offset=10):
    # Apply adaptive thresholding
    thresh1 = threshold_local(img1, block_size, offset=offset)
    binary1 = img1 < thresh1  # foreground is True

    thresh2 = threshold_local(img2, block_size, offset=offset)
    binary2 = img2 < thresh2  # foreground is True

    # Create 3-channel RGB output
    H, W = img1.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    red = binary1 & ~binary2
    blue = binary2 & ~binary1
    cyan = binary1 & binary2

    out[red] = [255, 0, 0]
    out[blue] = [0, 0, 255]
    out[cyan] = [0, 255, 255]

    return out


if __name__ == "__main__":
    app()
