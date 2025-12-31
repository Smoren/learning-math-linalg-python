import numpy as np

from app.analysers import EchelonMatrixAnalyser
from app.operations import add_matrices, mul_matrix, mul_matrices


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


if __name__ == '__main__':
    test_analysers()
    # example_transform_matrix_add_row()
    # example_transform_matrix_mul_row()
    # example_transform_matrix_swap_rows()
    # example_multiply_per_block()
