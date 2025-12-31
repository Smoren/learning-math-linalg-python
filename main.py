import numpy as np

from app.analysers import MatrixAnalyser, EchelonMatrixAnalyser, LinearSystemAnalyser, SquareEchelonMatrixAnalyser
from app.determinant import get_determinant
from app.examples import example_transform_matrix_add_row, example_transform_matrix_swap_rows, \
    example_transform_matrix_mul_row, example_multiply_per_block
from app.operations import add_matrices, mul_matrix, mul_matrices
from app.solvers import get_inverse_matrix, get_left_inverse_matrix, get_right_inverse_matrix
from app.system import LinearSystem
from app.transformers import LinearSystemGaussTransformer


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


def test_analysers():
    # A = np.array([
    #     [1, 2, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1],
    #     [0, 0, 0, 0],
    # ], dtype=np.float64)
    A = np.array([
        [1, 2, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float64)

    analyser = EchelonMatrixAnalyser(A)
    # print(f'det = {np.src.det(A)}')
    # print(analyser.is_square())
    # print(analyser.is_singular())
    # print(analyser.is_echelon())
    # print(analyser.is_reduced_echelon())
    print(analyser.is_reduced_echelon())


def test_operations():
    a = np.array([
        [1, 2],
        [3, 4],
    ])
    b = np.array([
        [5, 6],
        [7, 8],
    ])
    print(add_matrices(a, b))
    print(mul_matrix(a, 2))

    a = np.array([
        [1, 2],
        [4, 5],
        [7, 8],
    ])
    b = np.array([
        [5, 6, 7, 8],
        [7, 8, 9, 10],
    ])
    print(mul_matrices(a, b))


def test_linear_system_analyser():
    A = np.array([
        [0, 2, 0, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 8],
        [0, 0, 1, 0],
    ], dtype=np.float64)
    B = np.array([
        [1],
        [2],
        [3],
        [4],
    ], dtype=np.float64)
    linear_system = LinearSystem(A, B)

    # transformer = LinearSystemGaussTransformer(linear_system)
    # transformer.apply_gauss()
    # print(linear_system)

    analyser = LinearSystemAnalyser(linear_system)

    B_expected = np.array([
        [0.5],
        [0.5],
        [4],
        [0.375],
    ], dtype=np.float64)
    print(analyser.is_solution(B_expected))


def test_square_echelon_matrix_analyser():
    A = np.array([
        [0, 2, 3, 4],
        [4, 0, 5, 1],
        [5, 6, 0, 8],
        [6, 5, 1, 0],
    ], dtype=np.float64)
    B = np.array([
        [1],
        [2],
        [3],
        [4],
    ], dtype=np.float64)
    linear_system = LinearSystem(A, B)

    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()
    print(linear_system)

    analyser = SquareEchelonMatrixAnalyser(linear_system.A)
    print(analyser.is_invertible())
    print()


def test_solve_big_system():
    A = np.array([
        [0, 2, 3, 4],
        [4, 0, 5, 1],
        [5, 6, 0, 8],
        [6, 5, 1, 0],
    ], dtype=np.float64)
    B = np.eye(4, 4)

    linear_system = LinearSystem(A, B)
    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()
    print(linear_system)
    print()
    print(np.round(linear_system.B @ A))


def test_get_inverse_matrix():
    A = np.array([
        [0, 2, 3, 4],
        [4, 0, 5, 1],
        [5, 6, 0, 8],
        [6, 5, 1, 0],
    ], dtype=np.float64)
    Ai = get_inverse_matrix(A)

    print("A:")
    print(A)
    print()

    print("A^-1:")
    print(Ai)
    print()

    print("A^-1 * A:")
    print(np.round(Ai @ A, 8))
    print()

    print("A * A^-1:")
    print(np.round(A @ Ai, 8))
    print()


def test_get_left_inverse_matrix():
    A = np.array([
        [0, 2, 3],
        [4, 0, 5],
        [5, 6, 0],
        [6, 5, 1],
    ], dtype=np.float64)
    L = get_left_inverse_matrix(A)

    print("A:")
    print(A)
    print()

    print("L:")
    print(L)
    print()

    print("L * A:")
    print(np.round(L @ A, 8))
    print()


def test_get_right_inverse_matrix():
    A = np.array([
        [0, 2, 3, 4],
        [4, 0, 5, 1],
        [5, 6, 0, 8],
    ], dtype=np.float64)
    R = get_right_inverse_matrix(A)

    print("A:")
    print(A)
    print()

    print("R:")
    print(R)
    print()

    print("A * R:")
    print(np.round(A @ R, 8))
    print()


if __name__ == '__main__':
    test_determinant()
    # test_gauss()
    # test_analysers()
    # test_operations()
    # test_linear_system_analyser()
    # example_transform_matrix_add_row()
    # example_transform_matrix_mul_row()
    # example_transform_matrix_swap_rows()
    # example_multiply_per_block()
    # test_square_echelon_matrix_analyser()
    # test_solve_big_system()
    # test_get_inverse_matrix()
    # test_get_left_inverse_matrix()
    # test_get_right_inverse_matrix()
