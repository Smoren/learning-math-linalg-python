import numpy as np

from linalg.system import LinearSystem


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

    def _is_zero(self, x):
        return np.abs(x) <= np.finfo(self._linear_system.A.dtype).eps


class LinearSystemSquareGaussTransformer(LinearSystemBaseTransformer):
    def apply_gauss(self) -> "LinearSystemSquareGaussTransformer":
        if self._is_zero(np.linalg.det(self._linear_system.A)):
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
            if not self._is_zero(self._linear_system.A[i, i]):
                continue
            found = False
            for j in range(self._linear_system.A.shape[1]):
                if j != i and not self._is_zero(self._linear_system.A[j, i]):
                    self.add_rows(j, i, 1)
                    found = True
                    break
            if not found:
                raise ValueError(f"No allowed rows for column {i}")

    def _apply_gauss_forward(self):
        n = self._linear_system.A.shape[0]

        for i in range(n):
            # Если диагональный элемент равен нулю
            if self._is_zero(self._linear_system.A[i, i]):
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
            if not self._is_zero(self._linear_system.A[i, row_index]):
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
    pass
