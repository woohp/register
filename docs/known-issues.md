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

## Thin-plate spline

TPS produced an all-black image until the IRLS weighting fix; it now registers reasonably (Dice about 0.31 on both pages). Every match is a control point, so each IRLS step solves a dense (N+3)-square system, and the warp costs O(H·W·N). The `wendland-transform` branch explored a compactly supported alternative.

## No detection of failed registrations

A bad registration is returned like a good one. Options, cheapest first:

1. Reject near-collinear (degenerate) RANSAC samples before estimating; they are the main source of extreme-shear hypotheses.
2. Reject implausible hypotheses outright, e.g. a large singular-value ratio of the linear part, or Bernstein warps that fold (non-positive Jacobian determinant somewhere on the page).
3. Report a confidence such as the fraction of matches the final fit explains, so callers can reject.

## Warps do not anti-alias

OpenCV's `warpAffine`, `warpPerspective` and `remap` sample bilinearly. When the image has more pixels per template pixel than one, pre-shrink it with `cv2.resize(..., interpolation=cv2.INTER_AREA)` before warping. One global factor is exact for affine-family models and approximate for perspective, Bernstein and TPS. The test pages are the same size, so this does not show up there.
