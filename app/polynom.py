import numpy as np


class MatrixPolynom:
    _coefficients: np.ndarray

    def __init__(self, coefficients: np.ndarray):
        assert coefficients.ndim == 1
        self._coefficients = coefficients

    def __call__(self, X: np.ndarray):
        # return np.polyval(np.array(self._coefficients), X)
        result = np.zeros_like(X)

        for i in range(len(self._coefficients)):
            result += self._coefficients[i] * X**i

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
