import numpy as np

import register.transforms


def test_estimate_rigid_transform() -> None:
    # construct a rigid transformation matrix
    theta = 1.234
    tx = -3
    ty = 5
    T = np.array(
        [
            [tx, np.cos(theta), -np.sin(theta)],
            [ty, np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    ).T

    # create some dummy data, and then apply the transformation to the data
    A = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [1, -1, 0],
            [1, 0, -1],
        ],
        dtype=np.float32,
    )
    A[:, 0] = 1
    Y = A @ T

    # the estimated transformation should be the same as the true one
    estimated_T = register.transforms.RigidTransform.estimate(A, Y, 0.0)
    assert estimated_T.shape == (3, 2)
    np.testing.assert_array_almost_equal(estimated_T, T, 6)
    assert estimated_T.dtype == np.float32

    # it should also work for batched inputs
    A = np.stack([A] * 5, axis=0)
    Y = np.stack([Y] * 5, axis=0)
    estimated_T = register.transforms.RigidTransform.estimate(A, Y, 0.0)
    assert estimated_T.shape == (5, 3, 2)
    np.testing.assert_array_almost_equal(estimated_T, T[None].repeat(len(A), axis=0), 6)
    assert estimated_T.dtype == np.float32


def test_estimate_similarity_transform() -> None:
    # construct a rigid transformation matrix
    theta = 1.234
    tx = -3
    ty = 5
    scale = 0.123
    T = np.array(
        [
            [tx, np.cos(theta) * scale, -np.sin(theta) * scale],
            [ty, np.sin(theta) * scale, np.cos(theta) * scale],
        ],
        dtype=np.float32,
    ).T

    # create some dummy data, and then apply the transformation to the data
    A = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [1, -1, 0],
            [1, 0, -1],
        ],
        dtype=np.float32,
    )
    A[:, 0] = 1
    Y = A @ T

    # the estimated transformation should be the same as the true one
    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    np.testing.assert_array_almost_equal(estimated_T, T, 6)
    assert estimated_T.dtype == np.float32

    # it should also work for batched inputs
    A = np.stack([A] * 5, axis=0)
    Y = np.stack([Y] * 5, axis=0)
    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    assert estimated_T.shape == (5, 3, 2)
    np.testing.assert_array_almost_equal(estimated_T, T[None].repeat(len(A), axis=0), 6)
    assert estimated_T.dtype == np.float32

