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

    def swap_rows(self, index_from: int, index_to: int) -> "LinearSystemBaseTransformer":
        self._check_row_index_pair(index_from, index_to)

        self._linear_system.A[[index_from, index_to]] = self._linear_system.A[[index_to, index_from]]
        self._linear_system.B[[index_from, index_to]] = self._linear_system.B[[index_to, index_from]]

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
        row_index = 0
        pivots: List[Tuple[int, int]] = []
        for col_index in range(self._linear_system.A.shape[1]):
            pivot_row_index = self._find_row_index_of_max_non_zero_column(col_index, row_index)
            if pivot_row_index is None:
                continue

            if pivot_row_index != row_index:
                self.swap_rows(pivot_row_index, row_index)

            pivot = self._linear_system.A[row_index, col_index]
            for j in range(row_index + 1, self._linear_system.A.shape[0]):
                mult = -self._linear_system.A[j, col_index] / pivot
                self.add_rows(row_index, j, mult)

            if not is_zero(self._linear_system.A[row_index, col_index]):
                self.mul_row(row_index, 1/pivot)
                pivots.append((row_index, col_index))

            row_index += 1
            if row_index == self._linear_system.A.shape[0]:
                break

        for row_index, col_index in reversed(pivots):
            pivot = self._linear_system.A[row_index, col_index]
            for j in range(row_index-1, -1, -1):
                mult = -self._linear_system.A[j, col_index] / pivot
                self.add_rows(row_index, j, mult)

        return self

    def _find_row_index_of_max_non_zero_column(self, column_index: int, start_row_index: int) -> Optional[int]:
        candidates = self._linear_system.A[start_row_index:]
        if np.all(is_zero(candidates[:, column_index])):
            return None

        return int(np.argmax(np.abs(candidates[:, column_index]))) + start_row_index
