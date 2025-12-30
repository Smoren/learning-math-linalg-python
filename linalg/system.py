import numpy as np


class LinearSystem:
    """
    Система линейных алгебраических уравнений Ax = b.
    A - матрица коэффициентов, B - вектор свободных членов.

    Решение через обратную матрицу:
    (A^-1 * A) * x = A^-1 * b => Ex = A^-1 * b => x = A^-1 * b
    """

    _AB: np.ndarray
    _left_columns_count: int

    def __init__(self, A: np.ndarray, B: np.ndarray):
        assert A.ndim == 2 and B.ndim == 2
        assert A.shape[0] == B.shape[0] and A.shape[1] > 0 and B.shape[1] > 0

        self._AB = np.column_stack((A.copy(), B.copy()))
        self._left_columns_count = A.shape[1]

    def copy(self) -> "LinearSystem":
        return LinearSystem(self.A.copy(), self.B.copy())

    @property
    def A(self):
        return self._AB[:, :self._left_columns_count]

    @property
    def B(self):
        return self._AB[:, self._left_columns_count:]

    @property
    def AB(self):
        return self._AB

    @A.setter
    def A(self, value: np.ndarray):
        assert value.ndim == 2 and value.shape[0] == self._AB.shape[0] and value.shape[1] == self._left_columns_count
        self._AB[:, :value.shape[1]] = value

    @B.setter
    def B(self, value: np.ndarray):
        assert value.ndim == 2 and value.shape[0] == self._AB.shape[0] and value.shape[1] == self._AB.shape[1] - self._left_columns_count
        self._AB[:, self._left_columns_count:] = value

    @AB.setter
    def AB(self, value: np.ndarray):
        assert value.ndim == 2 and value.shape[0] == self._AB.shape[0] and value.shape[1] == self._AB.shape[1]
        self._AB = value

    def __repr__(self) -> str:
        all_strings = []

        for i in range(self.A.shape[0]):
            all_strings.append(" ".join(map(lambda x: f"{x:10.6g}", self.A[i].tolist())) + " | " + " ".join(map(lambda x: f"{x:10.6g}", self.B[i].tolist())))

        return "\n".join(all_strings)
