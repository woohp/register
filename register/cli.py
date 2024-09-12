from typing import Annotated

import cv2
from typer import Option, Typer

from register import generate_warped_image, register_image

app = Typer()


@app.command()
def main(
    image_path: str,
    template_path: str,
    method: str = "perspective",
    out_path: Annotated[str, Option("--out")] = "registered.png",
) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    T, score = register_image(image, template, method)
    print("score:", score)
    print("T:")
    print(T)

    registered_image = generate_warped_image(image, template.shape, T)
    cv2.imwrite(out_path, registered_image)


if __name__ == "__main__":
    app()
