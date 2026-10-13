# register agent guide

`register` aligns a photographed or scanned page (the *image*) to a *template* by matching BRISK/TEBLID features and robustly fitting a geometric transform (RANSAC, then IRLS with a Geman–McClure loss). It is a small library plus the `register-image` CLI.

Target Python 3.14 and current dependency releases. When dependency work is in scope, fix breaking changes rather than pinning old versions; do not turn unrelated tasks into upgrades.

## Workflow

- `make lint` (ruff + mypy) and `pytest` must pass; `make format` applies safe fixes.
- `tests/test_register.py` deliberately writes `registered_<method>.png` into the working directory so results can be inspected visually.
- Prefer outcome-based tests: recover a known transform from synthetic points, and judge real-image registration by ink overlap with the template. Seed randomness with a local `np.random.default_rng`.

## Fitting invariants

- All fitting happens in coordinates normalized by each image's longest side: `register_image` resizes to a longest side of `TARGET_MAX_SIZE` and divides keypoints by it. `warp` converts back to pixels.
- Transforms map **template → image** (`fit(template_points, image_points)`) because warping is inverse mapping: every template pixel samples the image.
- `fit` expects correspondences sorted by decreasing match quality. RANSAC's sampler (`utils/sampling.py`) is PROSAC-style and draws early hypotheses from a growing prefix.
- `Transform` methods are batched: RANSAC estimates and scores every hypothesis at once, so `expand`, `estimate`, `transform_points`, and `penalty` must accept arbitrary leading batch dimensions. Keep them vectorized.
- Penalties measure distortion beyond a similarity (shear, stretch, bending, perspective terms). They exist to tie-break and tame extreme candidates: RANSAC fits samples without them and applies them when scoring, and IRLS folds them into each fit. Never charge for rotation, uniform scale, or translation: documents are legitimately photographed sideways, nearer or farther, and off-center.
- `estimate` solves weighted least squares with explicit per-point weights. IRLS passes the Geman–McClure derivative normalized the same way as `_loss`, so each step minimizes a quadratic majorizer of the reported loss. Keep `estimate`'s objective consistent with `_loss` and `penalty` when changing regularization.

## Numerics

- Arrays are float32. numpy's type stubs often widen results to float64; use `cast(f32Array, ...)` instead of changing runtime dtypes.
- Use numba only where numpy would allocate large temporaries (RANSAC distance batches, the Hamming matrix). Decorate with `numba_njit` from `register/utils` using explicit signatures and `cache=True`. Only the outermost `prange` runs in parallel, so write inner loops with `range`.
- OpenCV's `warpAffine`, `warpPerspective`, and `remap` have no area interpolation (`INTER_AREA` silently becomes bilinear), so warps do not anti-alias when they downsample.
