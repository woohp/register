# Image Registration

This library provides tools for image registration through feature matching.
It supports perspective, affine, and polynomial transformations.


## Usage

Use as a library

```python
import register

image = cv2.imread('image.png', 0)  # only works with grayscale images
template = cv2.imread('template.png', 0)

T, scores = register.register_image(image, template)  # defaults to perspective transformation

# or for affine or quadratic transformations
T, scores = register.register_image(image, template, "affine")
T, scores = register.register_image(image, template, "quadratic")

# apply the transformation
registered_image = register.generate_warped_image(image, template.shape, T)
```

Use as a CLI tool

```bash
register-image image.png template.png --out registered.png  # defaults to perspective
register-image image.png template.png --method affine --out registered.png
register-image image.png template.png --method quadratic --out registered.png
```
