# Image Registration

This library provides tools for image registration through feature matching.
It supports perspective, affine, rigid, similarity, anisotropic similarity,
Bernstein polynomial, and thin-plate spline transformations.

## Usage

Use as a library

```python
import cv2
import register
from register.transforms import AffineTransform, PerspectiveTransform

image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)

# Register with a perspective transform (default)
transform = PerspectiveTransform()
register.register_image(image, template, transform)

# Apply the transformation
registered = register.generate_warped_image(image, template.shape, transform)
cv2.imwrite("registered.png", registered)

# Other transform types:
#   AffineTransform, RigidTransform, SimilarityTransform,
#   AnisotropicSimilarityTransform, BernsteinTransform(degrees=2)
```

Use as a CLI tool

```bash
register-image image.png template.png --out registered.png             # defaults to perspective
register-image image.png template.png --method affine --out registered.png
register-image image.png template.png --method bernstein -d 2 --out registered.png
```

Available methods: `perspective`, `affine`, `rigid`, `similarity`, `anisotropic_similarity`, `bernstein`, `thin_plate_spline`
