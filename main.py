import numpy as np

from linalg.analyzers import MatrixBaseAnalyser
from linalg.determinant import get_determinant
from linalg.system import LinearSystem
from linalg.transformers import LinearSystemGaussTransformer


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
    # A = np.array([
    #     [1, 2, 3, 0],
    #     [2, 4, 5, 0],
    #     [0, 0, 0, 1],
    #     [1, -4, 7, 0],
    # ], dtype=np.float64)
    # A = np.array([
    #     [1, 0, 3, 0],
    #     [2, 0, 5, 0],
    #     [0, 0, 0, 1],
    #     [1, 0, 7, 0],
    # ], dtype=np.float64)

    B = np.array([
        [1],
        [2],
        [3],
        [4],
    ], dtype=np.float64)

    linear_system = LinearSystem(A, B)
    print(linear_system)
    print()

    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()
    print(linear_system)
    print()

    print(f'det = {np.linalg.det(A)}')


def test_analyzers():
    A = np.array([
        [1, 2, 3, 4],
        [0, 0, 5, 1],
        [0, 0, 0, 8],
        # [0, 0, 0, 2],
    ], dtype=np.float64)

    analyser = MatrixBaseAnalyser(A)
    # print(f'det = {np.linalg.det(A)}')
    # print(analyser.is_square())
    # print(analyser.is_singular())
    print(analyser.is_echelon())
    # print(analyser.is_reduced_echelon())


if __name__ == '__main__':
    # test_determinant()
    # test_gauss()
    test_analyzers()

