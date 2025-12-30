import numpy as np

from linalg.analyzers import SquareMatrixAnalyser
from linalg.system import LinearSystem
from linalg.transformers import LinearSystemGaussTransformer


def get_inverse_matrix(A: np.ndarray):
    """Находит обратную матрицу для квадратной матрицы"""
    # Проверяем, что матрица квадратная (2D и размерности равны)
    assert A.ndim == 2 and A.shape[0] == A.shape[1]

    # Создаем единичную матрицу того же размера, что и исходная
    # Обратная матрица A^(-1) определяется как A * A^(-1) = E,
    # где E - единичная матрица
    E = np.eye(A.shape[0], A.shape[0])

    # Создаем линейную систему: A * X = E, где X - искомая обратная матрица
    # system.A = исходная матрица, system.B = единичная матрица
    system = LinearSystem(A, E)

    # Применяем метод Гаусса для приведения матрицы A к единичной
    # Одновременно преобразуем правую часть (единичную матрицу)
    transformer = LinearSystemGaussTransformer(system)
    transformer.apply_gauss()  # После этой операции A должна стать единичной матрицей

    # Проверяем, что исходная матрица была невырожденной (обратимой)
    # Если после метода Гаусса A стала единичной матрицей - матрица обратима
    analyzer = SquareMatrixAnalyser(system.A)
    if not analyzer.is_identity():
        # Матрица вырожденная (определитель = 0), обратной не существует
        raise ValueError('Input matrix is singular.')

    # После преобразований system.B содержит обратную матрицу,
    # так как мы решили уравнение A * X = I и получили X = A^(-1)
    return system.B
