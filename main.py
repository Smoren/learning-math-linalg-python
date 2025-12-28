import numpy as np

from linalg.determinant import get_determinant
from linalg.gauss import LinearSystem, LinearSystemInplaceTransformer


def test_determinant():
    a = np.array([
        [1, 2, 3, 4],
        [4, 3, 5, 1],
        [5, 6, 7, 8],
        [6, 5, 1, 2],
    ])

    a = np.random.rand(5, 5)

    actual = get_determinant(a)
    expected = np.linalg.det(a)

    print(actual, expected, abs(expected - actual))


def test_gauss():
    A = np.array([
        [0, 2, 3, 4],
        [4, 0, 5, 1],
        [5, 6, 0, 8],
        [6, 5, 1, 0],
    ], dtype=np.float64)
    # A = np.array([
    #     [0, 2, 0, 0],
    #     [4, 0, 0, 0],
    #     [0, 0, 0, 8],
    #     [0, 0, 1, 0],
    # ], dtype=np.float64)
    # A = np.array([
    #     [0, 2, 0, 0],
    #     [4, 0, 1, 0],
    #     [0, 0, 0, 8],
    #     [4, 0, 1, 0],
    # ], dtype=np.float64)
    A = np.array([
        [1, 2, 3, 0],
        [2, 4, 5, 0],
        [0, 0, 0, 1],
        [1, -4, 7, 0],
    ], dtype=np.float64)

    B = np.array([
        [1],
        [2],
        [3],
        [4],
    ], dtype=np.float64)

    linear_system = LinearSystem(A, B)
    print(linear_system)
    print()

    transformer = LinearSystemInplaceTransformer(linear_system)
    transformer.apply_gauss()
    print(linear_system)


if __name__ == '__main__':
    # test_determinant()
    test_gauss()
