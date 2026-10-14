from enum import StrEnum
from typing import Annotated, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from typer import Option, Typer

from register import register_image
from register.transforms import (
    AffineTransform,
    AnisotropicSimilarityTransform,
    BernsteinTransform,
    PerspectiveTransform,
    RigidTransform,
    SimilarityTransform,
    Transform,
)

u8Array = NDArray[np.uint8]


class Method(StrEnum):
    perspective = "perspective"
    affine = "affine"
    rigid = "rigid"
    similarity = "similarity"
    anisotropic_similarity = "anisotropic_similarity"
    bernstein = "bernstein"


TRANSFORMS: dict[Method, type[Transform]] = {
    Method.perspective: PerspectiveTransform,
    Method.affine: AffineTransform,
    Method.rigid: RigidTransform,
    Method.similarity: SimilarityTransform,
    Method.anisotropic_similarity: AnisotropicSimilarityTransform,
}

app = Typer()


@app.command()
def main(
    image_path: str,
    template_path: str,
    method: Annotated[Method, Option("--method", "-m")] = Method.perspective,
    degrees: Annotated[int, Option("--degrees", "-d", help="Polynomial degree for the bernstein method.")] = 2,
    out_path: Annotated[str, Option("--out", "-o")] = "registered.png",
    blend_path: Annotated[
        str | None, Option("--blend", help="Also write a red/blue/cyan overlay of the result against the template.")
    ] = None,
) -> None:
    transform = BernsteinTransform(degrees=degrees) if method is Method.bernstein else TRANSFORMS[method]()

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    if template is None:
        raise ValueError(f"Could not read template from {template_path}")

    register_image(cast(u8Array, image), cast(u8Array, template), transform)
    print("matrix:")
    print(transform.matrix)

    registered_image = transform.warp(cast(u8Array, image), template.shape)
    cv2.imwrite(out_path, registered_image)

    if blend_path is not None:
        cv2.imwrite(blend_path, color_blend(registered_image, cast(u8Array, template)))


def color_blend(image: u8Array, template: u8Array, block_size: int = 35, offset: int = 10) -> u8Array:
    """Color foreground pixels (BGR): red where only `image` has ink, blue for only `template`, cyan for both."""
    image_ink = _foreground(image, block_size, offset)
    template_ink = _foreground(template, block_size, offset)

    out = np.zeros((*image.shape, 3), dtype=np.uint8)
    out[image_ink & ~template_ink] = (0, 0, 255)
    out[template_ink & ~image_ink] = (255, 0, 0)
    out[image_ink & template_ink] = (255, 255, 0)
    return out


def _foreground(image: u8Array, block_size: int, offset: int) -> NDArray[np.bool]:
    binary = cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, offset)
    return binary.astype(bool)


if __name__ == "__main__":
    app()
