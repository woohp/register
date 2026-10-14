# Image Registration

This library provides tools for image registration through feature matching.
It supports perspective, affine, rigid, similarity, anisotropic similarity,
and Bernstein polynomial transformations.

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
registered = transform.warp(image, template.shape)
cv2.imwrite("registered.png", registered)

# Other transform types:
#   AffineTransform, RigidTransform, SimilarityTransform,
#   AnisotropicSimilarityTransform, BernsteinTransform(degrees=2)
```

Use as a CLI tool (`pip install -e '.[cli]'`)

```bash
register-image image.png template.png --out registered.png             # defaults to perspective
register-image image.png template.png --method affine --out registered.png
register-image image.png template.png --method bernstein -d 2 --out registered.png
register-image image.png template.png --blend blend.png  # also write a red/blue/cyan ink overlay against the template
```

Available methods: `perspective`, `affine`, `rigid`, `similarity`, `anisotropic_similarity`, `bernstein`
