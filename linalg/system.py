import numpy as np


class LinearSystem:
    """
    Система линейных алгебраических уравнений Ax = b.
    A - матрица коэффициентов, B - вектор свободных членов.

    Решение через обратную матрицу:
    (A^-1 * A) * x = A^-1 * b => Ex = A^-1 * b => x = A^-1 * b
    """

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
