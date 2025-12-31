import numpy as np
import pytest

from app.analysers import MatrixAnalyser, EchelonMatrixAnalyser
from app.system import LinearSystem
from app.transformers import LinearSystemGaussTransformer
from tests.transformers.gauss.fixtures import data_provider_for_gauss_for_invertible, \
    data_provider_for_gauss_for_singular


@pytest.mark.parametrize("data", data_provider_for_gauss_for_invertible())
def test_gauss_for_invertible(data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """
    Тестирование метода Гаусса для решения СЛАУ с обратимой матрицей.

    Проверяет, что метод Гаусса корректно преобразует систему в улучшенный ступенчатый вид
    и находит правильное решение.

    Args:
        data: Кортеж из трех numpy массивов:
              - A: Матрица коэффициентов СЛАУ
              - B: Вектор свободных членов
              - X: Ожидаемый вектор решения
    """
    A, B, X = data

    linear_system = LinearSystem(A, B)

    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()

    # Проверка, что левая матрица приведена к единичной форме
    np.testing.assert_array_equal(linear_system.A, np.eye(A.shape[0], A.shape[1]))

    # Проверка, что решение совпадает с ожидаемым
    np.testing.assert_array_equal(linear_system.B, X)


@pytest.mark.parametrize("data", data_provider_for_gauss_for_singular())
def test_gauss_for_singular(data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """
    Тестирование метода Гаусса для решения СЛАУ с вырожденной матрицей.

    Проверяет, что метод Гаусса корректно преобразует систему в улучшенный ступенчатый вид.

    Args:
        data: Кортеж из трех numpy массивов:
              - A: Матрица коэффициентов СЛАУ
              - B: Вектор свободных членов
              - A': Улучшенный ступенчатый вид матрицы A
              - B': Матрица B после преобразований
    """
    A, B, A_, B_ = data

    linear_system = LinearSystem(A, B)

    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()

    # Проверка, что матрица приведена к ступенчатому виду
    assert MatrixAnalyser(linear_system.A).is_echelon()

    # Проверка, что матрица приведена к улучшенному ступенчатому виду
    assert EchelonMatrixAnalyser(linear_system.A).is_reduced_echelon()

    # Проверка, что левая матрица соответствует ожидаемому
    np.testing.assert_array_equal(linear_system.A, A_)

    # Проверка, что решение совпадает с ожидаемым
    np.testing.assert_array_equal(linear_system.B, B_)
