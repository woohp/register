# AGENTS.md

Coding agent instructions for the `register` image registration library.

## Build/Lint/Test Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_transforms.py

# Run a single test function
pytest tests/test_transforms.py::test_estimate_rigid_transform

# Run a single test with parametrization
pytest "tests/test_transforms.py::test_estimate_rigid_transform[None]"

# Run tests with verbose output
pytest -v tests/test_transforms.py

# Format code (fix safe issues only)
make format

# Format code (fix all issues)
make fformat

# Lint and typecheck
make lint

# Run ruff only
ruff check register tests

# Run mypy only
mypy register tests
```

## Project Structure

- `register/` - Main library code
  - `__init__.py` - Core registration functions (`register_image`, `generate_warped_image`)
  - `transforms.py` - Transformation classes (Perspective, Affine, Rigid, Similarity, Quadratic, ThinPlateSpline)
  - `cli.py` - Typer-based CLI entry point
  - `utils/` - Numba-compiled utility functions
- `tests/` - Pytest test suite

## Code Style Guidelines

### Formatting (ruff)

- Line length: 120 characters
- Indent: 4 spaces
- Target: Python 3.14
- Preview mode enabled for ruff

### Imports

Group imports in this order, separated by blank lines:

1. Standard library (including `from __future__ import annotations` if needed)
2. Third-party packages (numpy, cv2, pytest, etc.)
3. Local imports

Example:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from .utils import get_pairwise_distances_squared
```

### Type Annotations

- Use strict typing: `allow_untyped_defs = false`
- Use `NDArray[np.float32]` and `NDArray[np.uint8]` for numpy arrays
- Define type aliases for common types:

  ```python
  f32Array = NDArray[np.float32]
  u8Array = NDArray[np.uint8]
  ```

- Use `Literal` for constrained string choices:

  ```python
  method_type = Literal["perspective", "affine", "rigid", "similarity", "quadratic", "thin_plate_spline"]
  ```

- Use `TypeVar` for generic array types:

  ```python
  FloatingNDArray = TypeVar("FloatingNDArray", NDArray[np.float32], NDArray[np.float64])
  ```

- Use `cast()` when type checker needs help:

  ```python
  return cast(u8Array, cv2.resize(image, ...))
  ```

- Use `type: ignore` comments sparingly when external libraries lack type stubs

### Naming Conventions

- Functions: `snake_case` (e.g., `register_image`, `get_distance_squared`)
- Classes: `PascalCase` (e.g., `AffineTransform`, `ThinPlateSplineTransform`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `TARGET_MAX_SIZE`, `N_ITERS_RANSAC`)
- Private functions: No underscore prefix, just keep them module-level
- Type aliases: `PascalCase` (e.g., `f32Array`, `u8Array`, `method_type`)

### Error Handling

- Use `ValueError` with descriptive messages for invalid inputs
- Use `raise ValueError(f"...")` with f-strings

### Functions

- Keep functions focused and single-purpose
- Use descriptive parameter names even if verbose
- Document public functions with docstrings; private functions need not have them
- Return tuples for multiple values: `tuple[f32Array, float]`

### Numba JIT Functions

- Use `numba_njit` wrapper defined in `register/utils/__init__.py`
- Specify explicit signatures for better performance:

  ```python
  @numba_njit("f4[:, ::1](i4, i4, f4[:, ::1])", cache=True, fastmath=True, boundscheck=False, parallel=True)
  ```

- Use `numba.prange` for parallel loops

### Testing

- Use `pytest.mark.parametrize` for testing multiple scenarios
- Use fixtures defined in `conftest.py` for shared test data
- Use `np.testing.assert_array_almost_equal` for array comparisons
- Name tests descriptively: `test_<function>_<scenario>`

### Ruff Ignore Rules

The following ruff rules are ignored:

- `B028` - No explicit stacklevel
- `B905` - zip without strict
- `C408` - Unnecessary collection call
- `E203` - Whitespace before ':'
- `SIM102` - Nested if statements
- `SIM108` - Ternary operator

## Important Notes

- This project uses Python 3.14 features
- Test assets are located in `tests/assets/`
- The CLI entry point is `register-image` after installation
