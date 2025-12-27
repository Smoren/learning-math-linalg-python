from collections import defaultdict
from typing import List, Set, Dict

import numpy as np

from linalg.determinant import get_determinant


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

    def apply_pivoting(self) -> "LinearSystemInplaceTransformer":
        assert self._linear_system.A.shape[0] == self._linear_system.A.shape[1]

        for i in range(self._linear_system.A.shape[0]):
            if not self._is_zero(self._linear_system.A[i, i]):
                continue
            found = False
            for j in range(self._linear_system.A.shape[1]):
                if j != i and not self._is_zero(self._linear_system.A[i, j]):
                    self.add_rows(j, i, 1)
                    found = True
                    break
            if not found:
                raise ValueError(f"No allowed rows for column {i}")

        return self

    def apply_gauss(self) -> "LinearSystemInplaceTransformer":
        if self._is_zero(get_determinant(self._linear_system.A)):
            raise ValueError("Matrix is singular")

        self.apply_pivoting()
        self.apply_gauss_bottom()
        self.apply_gauss_top()

        for i in range(self._linear_system.A.shape[0]):
            self.mul_row(i, 1/self._linear_system.A[i, i])

        return self

    def apply_gauss_bottom(self) -> "LinearSystemInplaceTransformer":
        for i in range(self._linear_system.A.shape[0]):
            for j in range(i+1, self._linear_system.A.shape[0]):
                mult = -self._linear_system.A[j, i] / self._linear_system.A[i, i]
                self.add_rows(i, j, mult)

        return self

    def apply_gauss_top(self) -> "LinearSystemInplaceTransformer":
        for i in range(self._linear_system.A.shape[0]):
            for j in range(i+1, self._linear_system.A.shape[0]):
                i_, j_ = self._linear_system.A.shape[0] - i - 1, self._linear_system.A.shape[0] - j - 1
                mult = -self._linear_system.A[j_, i_] / self._linear_system.A[i_, i_]
                self.add_rows(i_, j_, mult)

        return self


    def _check_row_index(self, index: int) -> None:
        assert 0 <= index < self._linear_system.A.shape[0]

    def _check_row_index_pair(self, index_from: int, index_to: int) -> None:
        self._check_row_index(index_from)
        self._check_row_index(index_to)
        assert index_from != index_to

    def _is_zero(self, x: float) -> bool:
        return abs(x) <= np.finfo(self._linear_system.A.dtype).eps
