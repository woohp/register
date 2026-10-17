# Known issues and next steps

Shortcomings found while reviewing the fitting pipeline, with the evidence we had at the time. Most of them are blocked on the same thing: there is no evaluation dataset, only the two pages in `tests/assets/`. Numbers below are ink-mask Dice overlap with the template (what `tests/test_register.py` uses; an unregistered page scores ~0.03).

## Regularization strength is a placeholder

Every transform uses `reg = 0.005`. The penalty measures distance from the nearest similarity, and at that strength it barely affects which hypothesis wins. A sweep on the two pages suggested the models want different strengths:

| reg | affine (page1 / page2) | Bernstein 2 | Bernstein 3 |
|---|---|---|---|
| 0.005 | 0.111 / 0.103 | 0.436 / 0.402 | 0.464 / 0.409 |
| 0.02 | 0.111 / 0.103 | 0.437 / 0.402 | 0.480 / 0.429 |
| 1 | 0.118 / 0.137 | 0.367 / 0.364 | 0.366 / 0.353 |

Bernstein's penalty also charges for genuine perspective bending, so strong values hurt it. Per-model defaults (roughly 1 for affine, 0.02 for Bernstein) look plausible but were not adopted on the strength of two images. Revisit with a dataset.

## The loss is a proxy, and misspecified models exploit it

The robust loss is measured on sparse feature matches. When the model cannot represent the page (affine-family models on a photo with perspective), fits converge to one of several "least wrong" solutions. Lower loss does not imply better overlap: on page2, affine reaches a lower penalized loss but a worse overlap (0.103) than a more similarity-like solution (0.130). When the model fits the page (perspective, Bernstein), loss and overlap agree, so the practical remedy is choosing a model that can represent the page. Optimizing pixel alignment directly (e.g. `cv2.findTransformECC`) would be a different algorithm altogether.

## Perspective IRLS minimizes algebraic, not geometric, error

`PerspectiveTransform.estimate` is a weighted DLT: it minimizes the algebraic residual, which equals the projective depth times the geometric residual the loss measures. So unlike the linear transforms, its IRLS steps are not exact majorize-minimize steps; the early stop catches steps that raise the loss. The exact fix is a weighted Gauss-Newton/Levenberg-Marquardt step on the geometric residuals. The easy approximation, refitting with weights divided by depth², lowered the loss only marginally (0.27769 to 0.27740 on page1), worsened ink overlap (0.335 to 0.317; page2 0.307 to 0.288), and doubled IRLS time, so it was not adopted.

## Local deformation (TPS was removed)

`ThinPlateSplineTransform` used every match as a control point, so fitting cost O(N³) and warping O(H·W·N); it nearly interpolated inlier noise, and it registered worse than Bernstein degree 3 on the test pages (0.31 vs 0.47). It was removed. If local deformation (e.g. a page curling near the spine) is needed, use a fixed grid of control points instead, which is linear in its parameters and slots into `_LinearTransform` (similarity penalty, consistent IRLS, cost independent of N): B-spline free-form deformation (`bsplines` branch; reproduces the identity like Bernstein) or fixed-center Wendland RBFs (`wendland-transform` branch). Either has too many parameters for RANSAC's minimal samples, so it should start from a perspective or Bernstein fit and refine with IRLS. Both branches are unreviewed, agent-written sketches that predate the IRLS and API fixes (the `bsplines` one already has visible bugs, e.g. `grid_size` is read from the class, and 8-point RANSAC samples for about 400 parameters), so reimplement on `_LinearTransform` with tests rather than merging them.

B-splines extend `BernsteinTransform` directly: a clamped B-spline of degree d with no interior knots is the Bernstein basis, so Bernstein is the one-cell case. That gives the IRLS starting point exactly: fit Bernstein with RANSAC, then insert knots (Boehm's algorithm) to rewrite the same warp on the finer grid. The identity's control points sit at the Greville abscissae (knot averages; i/d for Bernstein), which generalizes `similarity_basis`, and each knot cell is a Bézier patch, so the Bernstein-coefficient fold test (see below) applies per cell. The new design question is locality: control points of cells with few or no matches are barely constrained, and the similarity penalty pulls them toward the global similarity rather than toward their neighbors. A bending-energy (or control-point second-difference) penalty would make empty regions follow their neighbors; it is quadratic, so it fits `estimate`'s closed form and stays consistent with `_loss`. Legendre polynomials were considered as an alternative to Bernstein: they condition AᵀA better (2.7× at degree 2, 8× at degree 3, negligible in float32 at these degrees), but have no control-point, convex-hull, or knot-insertion structure.

## Rejecting bad registrations is the caller's job

By design, callers reject registrations whose `loss_` is NaN or above their own threshold; a bad registration is otherwise passed along like a good one. `fit` sets `loss_` to inf when the warp is not a valid page mapping over the template keypoints' bounding box (it folds, mirrors, or crosses a perspective horizon). The check uses that bounding box because `fit` does not know the template's shape. It is exact for the affine family (constant Jacobian) and perspective (the Jacobian determinant is det(H)/w³ with w affine, so the box's corners decide). For Bernstein it samples the Jacobian determinant on a 16×16 grid with finite differences, so a fold narrower than the grid spacing passes; a guarantee would need bounds on the determinant polynomial (e.g. its Bernstein coefficients, with subdivision). Finite differences are also why perspective uses the analytic test: a step can jump across a horizon lying just inside the box.

## Warps do not anti-alias

OpenCV's `warpAffine`, `warpPerspective` and `remap` sample bilinearly. When the image has more pixels per template pixel than one, pre-shrink it with `cv2.resize(..., interpolation=cv2.INTER_AREA)` before warping. One global factor is exact for affine-family models and approximate for perspective and Bernstein. The test pages are the same size, so this does not show up there.
