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
            [1, 0.1, 1],
            [1, 1, 0.2],
            [1, -1, 0.3],
            [1, -0.4, -1],
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

    # if we add some scale to the transformation matrix, the rigid transformation can't handle it
    T[1:] *= 2
    Y = A @ T
    estimated_T = register.transforms.RigidTransform.estimate(A, Y, 0.0)
    assert not np.allclose(estimated_T, T, atol=1e-5)


def test_estimate_rigid_transform_batched() -> None:
    batch_size = 5
    tx = np.random.uniform(-10, 10, (batch_size,))
    ty = np.random.uniform(-10, 10, (batch_size,))
    thetas = np.random.uniform(-np.pi, np.pi, (batch_size,))
    T = np.array(
        [
            [tx, np.cos(thetas), -np.sin(thetas)],
            [ty, np.sin(thetas), np.cos(thetas)],
        ],
        dtype=np.float32,
    ).transpose(2, 1, 0)  # [batch-size, 3, 2]

    # it should also work for batched inputs
    A = np.random.randn(batch_size, 91, 3).astype(np.float32)
    A[..., 0] = 1
    Y = A @ T
    estimated_T = register.transforms.RigidTransform.estimate(A, Y, 0.0)
    assert estimated_T.shape == (5, 3, 2)
    np.testing.assert_array_almost_equal(estimated_T, T, 5)
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
            [1, 0.1, 1],
            [1, 1, 0.2],
            [1, -1, 0.3],
            [1, -0.4, -1],
        ],
        dtype=np.float32,
    )
    A[:, 0] = 1
    Y = A @ T

    # the estimated transformation should be the same as the true one
    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    np.testing.assert_array_almost_equal(estimated_T, T, 6)
    assert estimated_T.dtype == np.float32


def test_estimate_similarity_transform_batched() -> None:
    batch_size = 5
    tx = np.random.uniform(-10, 10, (batch_size,))
    ty = np.random.uniform(-10, 10, (batch_size,))
    thetas = np.random.uniform(-np.pi, np.pi, (batch_size,))
    scales = np.random.uniform(-10, 10, (batch_size,))
    T = np.array(
        [
            [tx, np.cos(thetas) * scales, -np.sin(thetas) * scales],
            [ty, np.sin(thetas) * scales, np.cos(thetas) * scales],
        ],
        dtype=np.float32,
    ).transpose(2, 1, 0)  # [batch-size, 3, 2]

    # it should also work for batched inputs
    A = np.random.randn(batch_size, 91, 3).astype(np.float32)
    A[..., 0] = 1
    Y = A @ T
    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    assert estimated_T.shape == (5, 3, 2)
    np.testing.assert_array_almost_equal(estimated_T, T, 5)
    assert estimated_T.dtype == np.float32


def test_aniostropic_similarity_transform() -> None:
    # construct a rigid transformation matrix
    theta = 1.234
    tx = -3
    ty = 5
    scale1 = 0.123
    scale2 = 0.456
    T = np.array(
        [
            [tx, np.cos(theta) * scale1, -np.sin(theta) * scale1],
            [ty, np.sin(theta) * scale2, np.cos(theta) * scale2],
        ],
        dtype=np.float32,
    ).T

    # create some dummy data, and then apply the transformation to the data
    A = np.array(
        [
            [1, 0.1, 1],
            [1, 1, 0.2],
            [1, -1, 0.3],
            [1, -0.4, -1],
        ],
        dtype=np.float32,
    )
    A[:, 0] = 1
    Y = A @ T

    # the estimated transformation should be the same as the true one
    estimated_T = register.transforms.SimilarityTransform.estimate(A, Y, 0.0)
    np.testing.assert_array_almost_equal(estimated_T, T, 6)
