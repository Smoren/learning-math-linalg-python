from typing import Optional, Tuple, List

import numpy as np

from linalg.system import LinearSystem
from linalg.utils import is_zero


class LinearSystemBaseTransformer:
    _linear_system: LinearSystem

    def __init__(self, linear_system: LinearSystem):
        self._linear_system = linear_system

    def add_rows(self, index_from: int, index_to: int, mult: float) -> "LinearSystemBaseTransformer":
        self._check_row_index_pair(index_from, index_to)

        self._linear_system.A[index_to] += mult * self._linear_system.A[index_from]
        self._linear_system.B[index_to] += mult * self._linear_system.B[index_from]

        return self

    def mul_row(self, index: int, mult: float) -> "LinearSystemBaseTransformer":
        self._check_row_index(index)
        assert mult != 0

        self._linear_system.A[index] *= mult
        self._linear_system.B[index] *= mult

        return self

    def swap_rows(self, lhs: int, rhs: int) -> "LinearSystemBaseTransformer":
        self._check_row_index_pair(lhs, rhs)

        self._linear_system.A[[lhs, rhs]] = self._linear_system.A[[rhs, lhs]]
        self._linear_system.B[[lhs, rhs]] = self._linear_system.B[[rhs, lhs]]

        return self

    def _check_row_index(self, index: int) -> None:
        assert 0 <= index < self._linear_system.A.shape[0]

    def _check_row_index_pair(self, index_from: int, index_to: int) -> None:
        self._check_row_index(index_from)
        self._check_row_index(index_to)
        assert index_from != index_to


class LinearSystemSquareGaussTransformer(LinearSystemBaseTransformer):
    def __init__(self, linear_system: LinearSystem):
        assert linear_system.A.shape[0] == linear_system.A.shape[1]
        super().__init__(linear_system)

    def apply_gauss(self) -> "LinearSystemSquareGaussTransformer":
        if is_zero(np.linalg.det(self._linear_system.A)):
            raise ValueError("Matrix is singular")

        # self._apply_pivoting()
        self._apply_gauss_forward()
        self._apply_gauss_backward()

        for i in range(self._linear_system.A.shape[0]):
            self.mul_row(i, 1/self._linear_system.A[i, i])

        return self

    def _apply_pivoting(self):
        assert self._linear_system.A.shape[0] == self._linear_system.A.shape[1]

        for i in range(self._linear_system.A.shape[0]):
            if not is_zero(self._linear_system.A[i, i]):
                continue
            found = False
            for j in range(self._linear_system.A.shape[1]):
                if j != i and not is_zero(self._linear_system.A[j, i]):
                    self.add_rows(j, i, 1)
                    found = True
                    break
            if not found:
                raise ValueError(f"No allowed rows for column {i}")

    def _apply_gauss_forward(self):
        n = self._linear_system.A.shape[0]

        for i in range(n):
            # Если диагональный элемент равен нулю
            if is_zero(self._linear_system.A[i, i]):
                self._fix_pivot_forward(i)

            pivot = self._linear_system.A[i, i]
            for j in range(i + 1, n):
                mult = -self._linear_system.A[j, i] / pivot
                self.add_rows(i, j, mult)

        return self

    def _apply_gauss_backward(self) -> "LinearSystemSquareGaussTransformer":
        n = self._linear_system.A.shape[0]

        for i in range(n-1, -1, -1):
            pivot = self._linear_system.A[i, i]
            for j in range(i-1, -1, -1):
                mult = -self._linear_system.A[j, i] / pivot
                self.add_rows(i, j, mult)

        return self

    def _fix_pivot_forward(self, row_index: int) -> None:
        n = self._linear_system.A.shape[0]

        # То среди следующих строк
        for i in range(row_index + 1, n):
            # ищем первую, где в соответствующем столбце не ноль
            if not is_zero(self._linear_system.A[i, row_index]):
                # Имеем право менять местами, так как в последующих строках уже должны быть нули по предыдущим столбцам
                self.swap_rows(row_index, i)
                break
        else:
            # Если не нашли, то матрица вырождена
            raise ValueError(f"No allowed rows for column {row_index}")


class LinearSystemUniversalGaussTransformer(LinearSystemBaseTransformer):
    # TODO implement
    """
    Улучшенный ступенчатый вид (реализовать)
    0 * 0 * 0 * | *
    0 0 1 * 0 * | *
    0 0 0 0 1 * | *
    0 0 0 0 0 0 | *

    Сколько решений (0, 1, бесконечно много)
    Выразить главные переменные через свободные (столбцы с 1 соответствуют главным переменным, с * соответствуют свободным переменным)
    """
    def apply_gauss(self) -> "LinearSystemUniversalGaussTransformer":
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
