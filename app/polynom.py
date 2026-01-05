import numpy as np


# TODO test it
class MatrixPolynom:
    _coefficients: np.ndarray

    def __init__(self, coefficients: np.ndarray):
        assert coefficients.ndim == 1
        self._coefficients = self._trim_zeros(coefficients)

    def __call__(self, X: np.ndarray):
        assert X.ndim >= 2 and X.shape[0] == X.shape[1]

        # return np.polyval(np.array(self._coefficients), X)

        X_powered = X.copy()
        result = self._coefficients[0] * np.eye(*X.shape)

        for i in range(1, len(self._coefficients)):
            result += self._coefficients[i] * X_powered
            X_powered @= X

        return result

    def __add__(self, other: "MatrixPolynom"):
        coefficients = np.zeros(max(self._coefficients.size, other._coefficients.size))
        coefficients[:self._coefficients.size] += self._coefficients
        coefficients[:other._coefficients.size] += other._coefficients
        return MatrixPolynom(coefficients)

    def __mul__(self, other: "MatrixPolynom"):
        m, n = self._coefficients.size, other._coefficients.size

        self_coefficients = np.zeros(m+n)
        self_coefficients[:m] = self._coefficients
        other_coefficients = np.zeros(m+n)
        other_coefficients[:n] = other._coefficients

        coefficients = np.zeros(m+n)
        for k in range(m+n):
            c = sum(self_coefficients[i] * other_coefficients[k-i] for i in range(k+1))
            coefficients[k] = c

        return MatrixPolynom(coefficients)

    def __str__(self):
        terms = []
        for i, coeff in enumerate(self._coefficients):
            if coeff != 0:
                if i == 0:
                    terms.append(f"{coeff}")
                elif i == 1:
                    terms.append(f"{coeff}X")
                else:
                    terms.append(f"{coeff}X^{i}")
        return " + ".join(terms) if terms else "0"

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients.copy()

    @staticmethod
    def _trim_zeros(arr: np.ndarray):
        """Удаляет незначащие нули с конца."""
        if len(arr) == 0:
            return arr
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] != 0:
                return arr[:i + 1]
        return np.array([], dtype=arr.dtype)
