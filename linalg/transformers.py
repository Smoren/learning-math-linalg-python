from typing import Optional, Tuple, List

import numpy as np

from linalg.factories import create_transform_matrix_add_row, create_transform_matrix_mul_row, \
    create_transform_matrix_swap_rows
from linalg.system import LinearSystem
from linalg.utils import is_zero


class LinearSystemBaseTransformer:
    _linear_system: LinearSystem

    def __init__(self, linear_system: LinearSystem):
        self._linear_system = linear_system

    def add_rows(self, index_from: int, index_to: int, mult: float) -> "LinearSystemBaseTransformer":
        """Прибавляет к строке с индексом index_to строку с индексом index_from, умноженную на mult."""
        self._check_row_index_pair(index_from, index_to)

        # Создаем матрицу элементарного преобразования
        transformation_matrix = create_transform_matrix_add_row(index_from, index_to, self._linear_system.A.shape[0], mult)

        # Применяем преобразование к матрице A и B с помощью левого матричного умножения левой и правой сторон линейной системы
        self._linear_system.A = transformation_matrix @ self._linear_system.A
        self._linear_system.B = transformation_matrix @ self._linear_system.B

        # Пример тривиальной реализации
        # self._linear_system.A[index_to] += mult * self._linear_system.A[index_from]
        # self._linear_system.B[index_to] += mult * self._linear_system.B[index_from]

        return self

    def mul_row(self, index: int, mult: float) -> "LinearSystemBaseTransformer":
        """Умножает строку с индексом index на mult."""
        self._check_row_index(index)
        assert mult != 0

        # Создаем матрицу элементарного преобразования
        transformation_matrix = create_transform_matrix_mul_row(index, mult, self._linear_system.A.shape[0])

        # Применяем преобразование к матрице A и B с помощью левого матричного умножения левой и правой сторон линейной системы
        self._linear_system.A = transformation_matrix @ self._linear_system.A
        self._linear_system.B = transformation_matrix @ self._linear_system.B

        # Пример тривиальной реализации
        # self._linear_system.A[index] *= mult
        # self._linear_system.B[index] *= mult

        return self

    def swap_rows(self, lhs_index: int, rhs_index: int) -> "LinearSystemBaseTransformer":
        """Меняет местами строки с индексами lhs_index и rhs_index."""
        self._check_row_index_pair(lhs_index, rhs_index)

        # Создаем матрицу элементарного преобразования
        transformation_matrix = create_transform_matrix_swap_rows(lhs_index, rhs_index, self._linear_system.A.shape[0])

        # Применяем преобразование к матрице A и B с помощью левого матричного умножения левой и правой сторон линейной системы
        self._linear_system.A = transformation_matrix @ self._linear_system.A
        self._linear_system.B = transformation_matrix @ self._linear_system.B

        # Пример тривиальной реализации
        # self._linear_system.A[[lhs_index, rhs_index]] = self._linear_system.A[[rhs_index, lhs_index]]
        # self._linear_system.B[[lhs_index, rhs_index]] = self._linear_system.B[[rhs_index, lhs_index]]

        return self

    def _check_row_index(self, index: int) -> None:
        """Проверяет, что индекс строки находится в допустимом диапазоне."""
        assert 0 <= index < self._linear_system.A.shape[0]

    def _check_row_index_pair(self, index_from: int, index_to: int) -> None:
        """Проверяет, что индексы строк находятся в допустимом диапазоне и не равны друг другу."""
        self._check_row_index(index_from)
        self._check_row_index(index_to)
        assert index_from != index_to


class LinearSystemGaussTransformer(LinearSystemBaseTransformer):
    """
    Преобразует систему в улучшенный ступенчатый вид
    0 * 0 * 0 * | *
    0 0 1 * 0 * | *
    0 0 0 0 1 * | *
    0 0 0 0 0 0 | *

    Сложность: O(m×n×min(m, n)) ~ O(n^3)

    TODO:
    Сколько решений (0, 1, бесконечно много)
    Выразить главные переменные через свободные (столбцы с 1 соответствуют главным переменным, с * соответствуют свободным переменным)
    """
    def apply_gauss(self) -> "LinearSystemGaussTransformer":
        # Индекс текущей строки для обработки
        row_index = 0

        # Список опорных элементов (строка, столбец)
        pivots: List[Tuple[int, int]] = []

        # Если матрица пустая, то ничего не делаем
        if self._linear_system.A.shape[0] == 0:
            return self

        # Прямой ход метода Гаусса (проходим по столбцам)
        for col_index in range(self._linear_system.A.shape[1]):
            # Ищем строку с максимальным по модулю ненулевым элементом в текущем столбце
            pivot_row_index = self._find_row_index_of_max_non_zero_column(col_index, row_index)

            # Если все элементы в столбце нулевые, переходим к следующему столбцу, row_index не меняется
            if pivot_row_index is None:
                continue

            # Если строка с опорным элементом не совпадает с текущей, поменяем их местами
            if pivot_row_index != row_index:
                self.swap_rows(pivot_row_index, row_index)

            # После перестановки получаем гарантированно ненулевой опорный элемент, сохраняем его индексы
            pivot = self._linear_system.A[row_index, col_index]

            # Обнуляем элементы под опорным
            for j in range(row_index + 1, self._linear_system.A.shape[0]):
                # Получаем множитель для обнуления элемента (pivot != 0, значит деление безопасно)
                mult = -self._linear_system.A[j, col_index] / pivot
                # Добавляем строку row_index к строке j с коэффициентом mult, таким образом обнуляем элемент
                self.add_rows(row_index, j, mult)

            # Делим строку на опорный элемент, чтобы получить 1 в опорном элементе
            self.mul_row(row_index, 1/pivot)

            # Сохраняем индексы опорного элемента
            pivots.append((row_index, col_index))

            # Переходим к следующей строке
            row_index += 1
            # Если достигли последней строки матрицы, выходим из цикла
            if row_index == self._linear_system.A.shape[0]:
                break

        # Обратный ход метода Гаусса (проходим только по столбцам с опорными элементами в обратном порядке)
        for row_index, col_index in reversed(pivots):
            # Обнуляем элементы над опорным
            for j in range(row_index-1, -1, -1):
                # Получаем множитель для обнуления элемента (pivot = 1, значит делить на него не нужно)
                mult = -self._linear_system.A[j, col_index]
                # Добавляем строку row_index к строке j с коэффициентом mult, таким образом обнуляем элемент
                self.add_rows(row_index, j, mult)

        return self

    def _find_row_index_of_max_non_zero_column(self, column_index: int, start_row_index: int) -> Optional[int]:
        # Берем подматрицу начиная с текущей строки
        candidates = self._linear_system.A[start_row_index:]

        # Если все элементы в столбце нулевые, подходящих строк нет, возвращаем None
        if np.all(is_zero(candidates[:, column_index])):
            return None

        # Ищем индекс строки с максимальным по модулю ненулевым элементом в столбце
        return int(np.argmax(np.abs(candidates[:, column_index]))) + start_row_index
