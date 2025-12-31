import numpy as np
import sympy as sp
import pytest

from app.analysers import LinearSystemAnalyser, MatrixAnalyser, SquareEchelonMatrixAnalyser
from app.system import LinearSystem
from app.transformers import LinearSystemGaussTransformer
from tests.transformers.gauss.fixtures import data_provider_for_gauss_for_invertible, \
    data_provider_for_gauss_for_singular


@pytest.mark.parametrize("data", data_provider_for_gauss_for_invertible())
def test_gauss_for_invertible(data: tuple[np.ndarray, np.ndarray]):
    """
    Тестирование метода Гаусса для решения СЛАУ с обратимой матрицей.

    Проверяет, что метод Гаусса корректно преобразует систему в улучшенный ступенчатый вид
    и находит правильное решение.

    Args:
        data: Кортеж из трех numpy массивов:
              - A: Матрица коэффициентов СЛАУ
              - B: Вектор свободных членов
    """
    A, B = data

    linear_system = LinearSystem(A, B)

    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()

    # Проверка, что матрица приведена к ступенчатому виду правильно (сравниваем с эталонным рещением sympy)
    expected_rref = np.array(sp.Matrix(np.column_stack((A, B))).rref()[0].tolist(), dtype=np.float64)
    np.testing.assert_array_almost_equal(linear_system.AB, expected_rref)

    # Проверка, что матрица приведена к ступенчатому виду
    assert LinearSystemAnalyser(linear_system).is_reduced_echelon()

    # Проверка, что левая матрица приведена к единичной форме
    np.testing.assert_array_equal(linear_system.A, np.eye(A.shape[0], A.shape[1]))

    # Проверка, что решение найдено корректно
    assert LinearSystemAnalyser(LinearSystem(A, B)).is_solution(linear_system.B)

    if MatrixAnalyser(A).is_square():
        # Для квадратных матриц проверим, что матрица обратима
        assert SquareEchelonMatrixAnalyser(linear_system.A).is_invertible()


@pytest.mark.parametrize("data", data_provider_for_gauss_for_singular())
def test_gauss_for_singular(data: tuple[np.ndarray, np.ndarray]):
    """
    Тестирование метода Гаусса для решения СЛАУ с вырожденной матрицей.

    Проверяет, что метод Гаусса корректно преобразует систему в улучшенный ступенчатый вид.

    Args:
        data: Кортеж из трех numpy массивов:
              - A: Матрица коэффициентов СЛАУ
              - B: Вектор свободных членов
    """
    A, B = data

    linear_system = LinearSystem(A, B)

    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()

    # Проверка, что система приведена к улучшенному ступенчатому виду
    assert LinearSystemAnalyser(linear_system).is_reduced_echelon()

    # Проверка, что матрица приведена к ступенчатому виду правильно (сравниваем с эталонным рещением sympy)
    expected_rref = np.array(sp.Matrix(np.column_stack((A, B))).rref()[0].tolist(), dtype=np.float64)
    np.testing.assert_array_almost_equal(linear_system.AB, expected_rref)

    if MatrixAnalyser(A).is_square():
        # Для квадратных матриц проверим, что матрица вырождена
        assert SquareEchelonMatrixAnalyser(linear_system.A).is_singular()
