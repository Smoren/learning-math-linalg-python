import numpy as np

from linalg.system import LinearSystem
from linalg.utils import is_zero


class MatrixBaseAnalyser:
    _matrix: np.ndarray

    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix

    def is_square(self) -> bool:
        """Проверяет, является ли матрица квадратной (NxN)."""
        return self._matrix.shape[0] == self._matrix.shape[1]

    def is_singular(self) -> bool:
        """Проверяет, является ли матрица вырожденной (определитель равен 0)."""
        return not self.is_square() or is_zero(np.linalg.det(self._matrix))

    def is_echelon(self) -> bool:
        """Проверяет, является ли матрица ступенчатой."""
        for i in range(self._matrix.shape[0]):
            for j in range(i):
                if not is_zero(self._matrix[i, j]):
                    return False
        return True

    def is_reduced_echelon(self) -> bool:
        """Проверяет, является ли матрица улучшенным ступенчатым видом."""
        if not self.is_echelon():
            return False
        for i in range(self._matrix.shape[0]):
            if not is_zero(self._matrix[i, i]):
                for j in range(i + 1, self._matrix.shape[1]):
                    if not is_zero(self._matrix[i, j]):
                        return False
        return True


class LinearSystemBaseAnalyser:
    _linear_system: LinearSystem
    _matrix_analyser: MatrixBaseAnalyser

    def __init__(self, linear_system: LinearSystem):
        self._linear_system = linear_system
        self._matrix_analyser = MatrixBaseAnalyser(linear_system.A)

    def is_homogeneous(self) -> bool:
        """Проверяет, является ли система однородной (B = 0)."""
        return np.all(is_zero(self._linear_system.B))

    def is_square(self) -> bool:
        """Проверяет, является ли система квадратной (NxN)."""
        return self._matrix_analyser.is_square()

    def is_singular(self) -> bool:
        """Проверяет, является ли система вырожденной (определитель A равен 0)."""
        return not self._matrix_analyser.is_singular()
