import numpy as np

from linalg.analyzers import SquareMatrixAnalyser
from linalg.system import LinearSystem
from linalg.transformers import LinearSystemGaussTransformer


def get_inverse_matrix(matrix: np.ndarray):
    """Находит обратную матрицу для квадратной матрицы"""
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

    eye = np.eye(matrix.shape[0], matrix.shape[0])
    system = LinearSystem(matrix, eye)

    transformer = LinearSystemGaussTransformer(system)
    transformer.apply_gauss()

    analyzer = SquareMatrixAnalyser(system.A)
    if not analyzer.is_identity():
        raise ValueError('Input matrix is singular.')

    return system.B
