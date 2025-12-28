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
        if not self.is_square():
            raise ValueError("Only square matrices can be singular")
        return is_zero(np.linalg.det(self._matrix))

    def is_echelon(self) -> bool:
        """Проверяет, является ли матрица ступенчатой."""
        # Получаем размеры матрицы
        rows, cols = self._matrix.shape

        # Обработка пустой матрицы: если нет строк или нет столбцов,
        # считаем матрицу ступенчатой (тривиальный случай)
        if rows == 0 or cols == 0:
            return True

        # Инициализируем счетчик нулей для предыдущей строки значением -1
        # Это гарантирует, что первая строка пройдет проверку (любое zero_count > -1)
        prev_zero_count = -1

        # Проходим по всем строкам матрицы сверху вниз
        for row_index in range(rows):
            # Счетчик ведущих нулей в текущей строке
            zero_count = 0

            # Подсчитываем количество нулей до первого ненулевого элемента в строке
            for col_index in range(cols):
                # Если встречаем ненулевой элемент, прерываем подсчет
                if not is_zero(self._matrix[row_index, col_index]):
                    break
                zero_count += 1  # Увеличиваем счетчик для каждого пройденного нуля

            # Проверка условия ступенчатости:
            # количество ведущих нулей в текущей строке должно быть СТРОГО БОЛЬШЕ, чем в предыдущей строке
            if zero_count <= prev_zero_count:
                # Если условие нарушено (<=), матрица не является ступенчатой
                return False

            # Сохраняем значение для следующей итерации (сравнения со следующей строкой)
            prev_zero_count = zero_count

        # Если все строки прошли проверку, матрица имеет ступенчатый вид
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
