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

        for row in self.A:
            all_strings.extend([f"{num:.6g}" if isinstance(num, float) else str(num) for num in row.tolist()])

        for num in self.B:
            all_strings.append(f"{num:.6g}" if isinstance(num, float) else str(num))

        max_width = max(len(s) for s in all_strings) + 2  # +2 для отступов

        result = []
        for i, row in enumerate(self.A):
            a_part = " ".join(
                [f"{num:{max_width}.6g}" if isinstance(num, float) else f"{num:>{max_width}}"
                 for num in row.tolist()]
            )

            b_num = self.B[i][0]
            b_str = f"{b_num:{max_width}.6g}" if isinstance(b_num, float) else f"{b_num:>{max_width}}"

            result.append(f"{a_part}  | {b_str}")

        return "\n".join(result)


class LinearSystemInplaceTransformer:
    _linear_system: LinearSystem

    def __init__(self, linear_system: LinearSystem):
        self._linear_system = linear_system

    def add_rows(self, index_from: int, index_to: int, mult: float) -> "LinearSystemInplaceTransformer":
        self._check_row_index_pair(index_from, index_to)
        assert mult != 0

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

    def _check_row_index(self, index: int) -> None:
        assert 0 <= index < self._linear_system.A.shape[0]

    def _check_row_index_pair(self, index_from: int, index_to: int) -> None:
        self._check_row_index(index_from)
        self._check_row_index(index_to)
        assert index_from != index_to
