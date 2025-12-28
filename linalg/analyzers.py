from typing import Set

import numpy as np

from linalg.operations import mul_matrices
from linalg.system import LinearSystem
from linalg.utils import is_zero


class MatrixAnalyser:
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

    def is_zero(self) -> bool:
        """Проверяет, является ли матрица нулевой (все элементы равны 0)."""
        return np.all(is_zero(self._matrix))

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

            # Пропускаем нулевые строки
            if np.all(is_zero(self._matrix[row_index])):
                # Если строка состоит только из нулей, то ведущих нулей столько же, сколько столбцов
                prev_zero_count = cols
                continue

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

    def is_inverse_left(self, another: np.ndarray) -> bool:
        """Проверяет, является ли матрица левым обратным для другой матрицы."""
        # Проверим, что матрицы можно перемножать
        if self._matrix.shape[1] != another.shape[0]:
            return False

        # Проверим, что результат будет квадратной матрицей
        if self._matrix.shape[0] != another.shape[1]:
            return False

        # Перемножим матрицы
        result = mul_matrices(self._matrix, another)

        # Проверим, что результат - единичная матрица
        return SquareMatrixAnalyser(result).is_identity()

    def is_inverse_right(self, another: np.ndarray) -> bool:
        """Проверяет, является ли матрица правым обратным для другой матрицы."""
        # Проверим, что матрицы можно перемножать
        if self._matrix.shape[1] != another.shape[0]:
            return False

        # Проверим, что результат будет квадратной матрицей
        if self._matrix.shape[0] != another.shape[1]:
            return False

        # Перемножим матрицы
        result = mul_matrices(another, self._matrix)

        # Проверим, что результат - единичная матрица
        return SquareMatrixAnalyser(result).is_identity()


class SquareMatrixAnalyser(MatrixAnalyser):
    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        if not self.is_square():
            raise ValueError("Matrix is not square")

    def is_diagonal(self) -> bool:
        """Проверяет, является ли матрица диагональной (имеет только диагональные элементы, остальные нули)."""
        for i in range(self._matrix.shape[0]):
            for j in range(self._matrix.shape[1]):
                # Если элемент не диагональный и не равен нулю, матрица не диагональная
                if i != j and not is_zero(self._matrix[i, j]):
                    return False
        return True

    def is_identity(self) -> bool:
        """Проверяет, является ли матрица единичной (диагональная, все диагональные элементы равны 1)."""
        if not self.is_diagonal():
            return False

        for i in range(self._matrix.shape[0]):
            if not np.isclose(self._matrix[i, i], 1):
                return False

        return True

    def is_scalar(self) -> bool:
        """Проверяет, является ли матрица скалярной (диагональная, все диагональные элементы равны между собой)."""
        if not self.is_diagonal():
            return False

        first_diagonal_element = self._matrix[0, 0]
        for i in range(self._matrix.shape[0]):
            if not np.isclose(self._matrix[i, i], first_diagonal_element):
                return False
        return True

    def is_inverse(self, another: np.ndarray) -> bool:
        """
        Проверяет, является ли матрица обратной для другой матрицы.
        Достаточно проверить только слева, потому что: LA = E, AR = E => L = R
        Доказательство: LAR = (LA)R = ER = R; LAR = L(AR) = LE = L => L = R
        """
        assert another.shape == self._matrix.shape
        result = mul_matrices(self._matrix, another)
        return SquareMatrixAnalyser(result).is_identity()

    def get_trace(self) -> float:
        """Возвращает след матрицы (сумма диагональных элементов)."""
        result = 0
        for i in range(self._matrix.shape[0]):
            result += self._matrix[i, i]
        return result


class EchelonMatrixAnalyser(MatrixAnalyser):
    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        if not self.is_echelon():
            raise ValueError("Matrix is not in echelon form")

    def get_pivot_columns(self) -> Set[int]:
        """Возвращает множество индексов столбцов, содержащих опорные элементы."""
        pivot_columns = set()

        last_col_index = 0
        for row_index in range(self._matrix.shape[0]):
            for col_index in range(last_col_index, self._matrix.shape[1]):
                if not is_zero(self._matrix[row_index, col_index]):
                    pivot_columns.add(col_index)
                    last_col_index = col_index
                    break
        return pivot_columns

    def get_rank(self) -> int:
        """Возвращает ранг матрицы (количество опорных элементов)."""
        return len(self.get_pivot_columns())

    def is_reduced_echelon(self) -> bool:
        """Проверяет, имеет ли матрица улучшенный (приведенный) ступенчатый вид."""
        # Получаем множество столбцов, содержащих опорные (ведущие) элементы
        # В ступенчатой матрице каждый такой столбец содержит ровно один опорный элемент
        pivot_columns = self.get_pivot_columns()

        # Проверяем каждый столбец с опорным элементом
        for col_index in pivot_columns:
            # Получаем все ненулевые элементы в текущем столбце
            non_zeros = self._matrix[:, col_index][~is_zero(self._matrix[:, col_index])]

            # Проверяем два условия RREF для текущего столбца:
            # 1. non_zeros.shape[0] != 1: в столбце должен быть ровно один ненулевой элемент
            #    (опорный элемент, все остальные должны быть нулями)
            # 2. np.isclose(non_zeros, 1).sum() != 1: этот единственный ненулевой элемент
            #    должен быть равен 1 (с учетом численной погрешности)
            if non_zeros.shape[0] != 1 or np.isclose(non_zeros, 1).sum() != 1:
                return False

        # Если все столбцы с опорными элементами прошли проверку,
        # матрица является приведенной ступенчатой (RREF)
        return True


class LinearSystemAnalyser:
    _linear_system: LinearSystem

    def __init__(self, linear_system: LinearSystem):
        self._linear_system = linear_system

    def is_homogeneous(self) -> bool:
        """Проверяет, является ли система однородной (B = 0)."""
        return np.all(is_zero(self._linear_system.B))

    def is_square(self) -> bool:
        """Проверяет, является ли система квадратной (NxN)."""
        return MatrixAnalyser(self._linear_system.A).is_square()

    def is_singular(self) -> bool:
        """Проверяет, является ли система вырожденной (определитель A равен 0)."""
        return not MatrixAnalyser(self._linear_system.A).is_singular()

    def is_echelon(self) -> bool:
        """Проверяет, является ли система ступенчатой."""
        return MatrixAnalyser(self._linear_system.A).is_echelon()

    def is_reduced_echelon(self) -> bool:
        """Проверяет, является ли система улучшенной (приведенной) ступенчатой."""
        return MatrixAnalyser(self._linear_system.A).is_echelon() and EchelonMatrixAnalyser(self._linear_system.A).is_reduced_echelon()

    def is_solution(self, X: np.ndarray) -> bool:
        """Проверяет, является ли вектор X решением системы."""
        # Проверяем, что X имеет один столбец, а количество строк совпадает с количеством столбцов матрицы A
        assert X.shape == (self._linear_system.A.shape[1], 1)

        # Вычисляем правую часть системы (A * X)
        B_actual = mul_matrices(self._linear_system.A, X)
        # Ожидаемая правая часть системы
        B_expected = self._linear_system.B

        # Проверяем, что правые части совпадают
        return np.all(is_zero(B_actual - B_expected))


class EchelonLinearSystemAnalyzer(LinearSystemAnalyser):
    def __init__(self, linear_system: LinearSystem):
        super().__init__(linear_system)
        if not MatrixAnalyser(linear_system.A).is_echelon():
            raise ValueError("Matrix is not in echelon form")

    def get_pivot_columns(self) -> Set[int]:
        """Возвращает множество индексов столбцов, содержащих опорные элементы."""
        return EchelonMatrixAnalyser(self._linear_system.A).get_pivot_columns()

    def get_rank(self) -> int:
        """Возвращает ранг системы (количество опорных элементов)."""
        return EchelonMatrixAnalyser(self._linear_system.A).get_rank()

    def is_reduced_echelon(self) -> bool:
        """Проверяет, является ли система улучшенной (приведенной) ступенчатой."""
        return EchelonMatrixAnalyser(self._linear_system.A).is_reduced_echelon()
