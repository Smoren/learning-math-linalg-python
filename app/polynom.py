import numpy as np

from app.utils import is_zero


# TODO test it
class MatrixPolynom:
    _coefficients: np.ndarray

    def __init__(self, coefficients: np.ndarray):
        assert coefficients.ndim == 1
        self._coefficients = self._normalize(coefficients)

    def __call__(self, X: np.ndarray):
        assert X.ndim >= 2 and X.shape[0] == X.shape[1]

        # Нативная реализация numpy
        # return np.polyval(np.array(self._coefficients), X)

        if self._coefficients.size == 0:
            return np.zeros(X.shape)

        X_powered = X.copy()
        result = self._coefficients[0] * np.eye(*X.shape)

        for i in range(1, len(self._coefficients)):
            result += self._coefficients[i] * X_powered
            X_powered @= X

        return result

    def __add__(self, other: "MatrixPolynom"):
        # Нативная реализация numpy
        # return MatrixPolynom(np.polynomial.polynomial.polyadd(self._coefficients, other._coefficients))

        coefficients = np.zeros(max(self._coefficients.size, other._coefficients.size), dtype=np.result_type(self._coefficients, other._coefficients))
        coefficients[:self._coefficients.size] += self._coefficients
        coefficients[:other._coefficients.size] += other._coefficients
        return MatrixPolynom(coefficients)

    def __mul__(self, other: "MatrixPolynom"):
        # Нативная реализация numpy
        # return MatrixPolynom(np.convolve(self._coefficients, other._coefficients))

        # Получаем степени полиномов
        # m = deg(P), n = deg(Q)
        m, n = self.degree, other.degree

        # Создаём расширенные массивы коэффициентов под размер массива коэффициентов результата произведения
        # Размер m+n+1 - максимальная степень произведения полиномов + свободный член
        # Например, для полиномов степени 2 и 3 результат имеет степень 5, нужно 6 коэффициентов
        self_coefficients = np.zeros(m+n+1, dtype=self._coefficients.dtype)
        other_coefficients = np.zeros(m+n+1, dtype=other._coefficients.dtype)

        # Копируем существующие коэффициенты в начало массивов, остальные позиции заполнены нулями
        # Это позволяет использовать простую формулу c_k = Σ a_i·b_{k-i} без проверок границ
        self_coefficients[:self._coefficients.size] = self._coefficients
        other_coefficients[:other._coefficients.size] = other._coefficients

        # Инициализируем массив для коэффициентов произведения
        coefficients = np.zeros(m+n+1, dtype=np.result_type(self._coefficients, other._coefficients))

        # Вычисляем свёртку коэффициентов
        for k in range(m+n+1):
            # Получаем коэффициент c_k при степени x^k:
            # c_k = ∑_{i+j=k} a_i*b_j, где i = 0...m, j = 0...n

            # Суммируем все пары (i, j) такие что i + j = k
            # i пробегает от 0 до k включительно, j автоматически равен k-i
            c = sum(self_coefficients[i] * other_coefficients[k-i] for i in range(k+1))
            coefficients[k] = c

        # Создаём новый полином с вычисленными коэффициентами
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
    def degree(self) -> int:
        if self._coefficients.size == 0:
            return 0
        return self._coefficients.size - 1

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients.copy()

    @staticmethod
    def _normalize(coefficients: np.ndarray):
        # Убираем незначащие нули с конца массива
        result = np.trim_zeros(coefficients, trim='b')

        # Возвращаем результат (если массив пустой, добавляем нуль)
        return result if len(result) > 0 else np.array([0.0], dtype=coefficients.dtype)
