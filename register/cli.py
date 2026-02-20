from typing import Annotated

import cv2
from typer import Option, Typer

from register import generate_warped_image, register_image
from register.transforms import method_type

app = Typer()


@app.command()
def main(
    image_path: str,
    template_path: str,
    method: method_type = "perspective",
    out_path: Annotated[str, Option("--out")] = "registered.png",
) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    if template is None:
        raise ValueError(f"Could not read template from {template_path}")

    T, score = register_image(image, template, method)  # type: ignore
    print("score:", score)
    print("T:")
    print(T)

    registered_image = generate_warped_image(image, template.shape, T)  # type: ignore
    cv2.imwrite(out_path, registered_image)


if __name__ == "__main__":
    app()
