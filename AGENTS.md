# register agent guide

`register` aligns a photographed or scanned page (the *image*) to a *template* by matching BRISK/TEBLID features and robustly fitting a geometric transform (RANSAC, then IRLS with a Geman–McClure loss). It is a small library plus the `register-image` CLI.

Target Python 3.14 and current dependency releases. When dependency work is in scope, fix breaking changes rather than pinning old versions; do not turn unrelated tasks into upgrades.

## Workflow

- `docs/known-issues.md` records known shortcomings, what the evidence showed, and next steps. Read it before tuning regularization, the loss, or matching, and update it when an item is resolved or new evidence arrives.
- `make lint` (ruff + mypy) and `pytest` must pass; `make format` applies safe fixes.
- `tests/test_register.py` deliberately writes `registered_<method>.png` into the working directory so results can be inspected visually.
- Prefer outcome-based tests: recover a known transform from synthetic points, and judge real-image registration by ink overlap with the template. Seed randomness with a local `np.random.default_rng`.

## Fitting invariants

- All fitting happens in coordinates normalized by each image's longest side, with pixel centers at integer pixel coordinates. `detect_features` detects on a copy resized to a longest side of `TARGET_MAX_SIZE`, maps keypoints back to original pixels (`cv2.resize` aligns pixel centers), then normalizes. `warp` converts back to pixels.
- Transforms map **template → image** (`fit(template_points, image_points)`) because warping is inverse mapping: every template pixel samples the image.
- Transforms follow scikit-learn conventions: constructors only store settings; `fit` returns `self` and sets `matrix_` and `loss_`. Callers reject bad registrations by thresholding `loss_` (inf marks invalid warps), so keep it meaningful and always set.
- `fit` expects correspondences sorted by decreasing match quality. RANSAC's sampler (`utils/sampling.py`) is PROSAC-style and draws early hypotheses from a growing prefix.
- `Transform` methods are batched: RANSAC estimates and scores every hypothesis at once, so `expand`, `estimate`, `transform_points`, `squared_residuals`, and `penalty` must accept arbitrary leading batch dimensions. Keep them vectorized; `squared_residuals` scores every hypothesis with one matrix product rather than materializing their transformed points (numpy's batched matmul over thousands of tiny matrices is slow).
- Penalties measure distortion beyond a similarity (shear, stretch, bending, perspective terms). They exist to tie-break and tame extreme candidates: RANSAC fits samples without them and applies them when scoring, and IRLS folds them into each fit. Never charge for rotation, uniform scale, or translation: documents are legitimately photographed sideways, nearer or farther, and off-center.
- `estimate` solves weighted least squares with explicit per-point weights. IRLS passes the Geman–McClure derivative normalized the same way as `_loss`, so each step minimizes a quadratic majorizer of the reported loss. Keep `estimate`'s objective consistent with `_loss` and `penalty` when changing regularization.

## Numerics

- Arrays are float32. numpy's type stubs often widen results to float64; use `cast(f32Array, ...)` instead of changing runtime dtypes.
- `_LinearTransform.squared_residuals` must stay float64 internally: it expands |aT − y|² into a matrix product, and in float32 the cancellation swamps inlier residuals.
- OpenCV's `warpAffine`, `warpPerspective`, and `remap` have no area interpolation (`INTER_AREA` silently becomes bilinear), so warps do not anti-alias when they downsample.
