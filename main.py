import numpy as np

from app.analysers import MatrixAnalyser, EchelonMatrixAnalyser, LinearSystemAnalyser, SquareEchelonMatrixAnalyser
from app.examples import example_transform_matrix_add_row, example_transform_matrix_swap_rows, \
    example_transform_matrix_mul_row, example_multiply_per_block
from app.operations import add_matrices, mul_matrix, mul_matrices
from app.solvers import get_inverse_matrix, get_left_inverse_matrix, get_right_inverse_matrix, get_determinant
from app.system import LinearSystem
from app.transformers import LinearSystemGaussTransformer


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


if __name__ == '__main__':
    test_analysers()
    # test_operations()
    # example_transform_matrix_add_row()
    # example_transform_matrix_mul_row()
    # example_transform_matrix_swap_rows()
    # example_multiply_per_block()
