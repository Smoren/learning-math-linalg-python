import numpy as np


class LinearSystem:
    A: np.ndarray
    B: np.ndarray

    def __init__(self, A: np.ndarray, B: np.ndarray):
        assert A.ndim == 2 and B.ndim == 2
        assert A.shape[0] == B.shape[0] and B.shape[1] == 1

        self.A = A
        self.B = B

    def copy(self) -> "LinearSystem":
        return LinearSystem(self.A.copy(), self.B.copy())

    def __repr__(self) -> str:
        all_strings = []

        for i in range(self.A.shape[0]):
            all_strings.append(" ".join(map(lambda x: f"{x:10.6g}", self.A[i].tolist())) + " | " + f"{self.B[i, 0]:10.6g}")

        return "\n".join(all_strings)


class LinearSystemInplaceTransformer:
    _linear_system: LinearSystem

    def __init__(self, linear_system: LinearSystem):
        self._linear_system = linear_system

    def add_rows(self, index_from: int, index_to: int, mult: float) -> "LinearSystemInplaceTransformer":
        self._check_row_index_pair(index_from, index_to)

        self._linear_system.A[index_to] += mult * self._linear_system.A[index_from]
        self._linear_system.B[index_to] += mult * self._linear_system.B[index_from]

        return self

    def mul_row(self, index: int, mult: float) -> "LinearSystemInplaceTransformer":
        self._check_row_index(index)
        assert mult != 0

        self._linear_system.A[index] *= mult
        self._linear_system.B[index] *= mult

        return self

    def swap_rows(self, index_from: int, index_to: int) -> "LinearSystemInplaceTransformer":
        self._check_row_index_pair(index_from, index_to)

        self._linear_system.A[[index_from, index_to]] = self._linear_system.A[[index_to, index_from]]
        self._linear_system.B[[index_from, index_to]] = self._linear_system.B[[index_to, index_from]]

        return self

    def apply_gauss(self) -> "LinearSystemInplaceTransformer":
        if self._is_zero(np.linalg.det(self._linear_system.A)):
            raise ValueError("Matrix is singular")

        # self.apply_pivoting()
        self.apply_gauss_forward()
        self.apply_gauss_backward()

        for i in range(self._linear_system.A.shape[0]):
            self.mul_row(i, 1/self._linear_system.A[i, i])

        return self

    def apply_pivoting(self) -> "LinearSystemInplaceTransformer":
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

        return self

    def apply_gauss_forward(self) -> "LinearSystemInplaceTransformer":
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

    def apply_gauss_backward(self) -> "LinearSystemInplaceTransformer":
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

    def _check_row_index(self, index: int) -> None:
        assert 0 <= index < self._linear_system.A.shape[0]

    def _check_row_index_pair(self, index_from: int, index_to: int) -> None:
        self._check_row_index(index_from)
        self._check_row_index(index_to)
        assert index_from != index_to

    def _is_zero(self, x: float) -> bool:
        return abs(x) <= np.finfo(self._linear_system.A.dtype).eps
