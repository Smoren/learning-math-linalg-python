import numpy as np
import pytest

from app.system import LinearSystem
from app.transformers import LinearSystemGaussTransformer
from tests.transformers.gauss.fixtures import data_provider_for_gauss


@pytest.mark.parametrize("data", data_provider_for_gauss())
def test_gauss_for_invertible(data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """
    Тестирование метода Гаусса для решения СЛАУ с обратимой матрицей.

    Проверяет, что метод Гаусса корректно преобразует систему в ступенчатый вид
    и находит правильное решение.

    Args:
        data: Кортеж из трех numpy массивов:
              - A: Матрица коэффициентов СЛАУ
              - B: Вектор свободных членов
              - X: Ожидаемый вектор решения

    Внутри теста:
        1. Создается объект LinearSystem с заданными матрицами A и B
        2. Создается трансформер LinearSystemGaussTransformer
        3. Применяется метод Гаусса
        4. Проверяется, что матрица A превратилась в единичную
        5. Проверяется, что вектор B совпадает с ожидаемым решением
    """
    A, B, X = data

    linear_system = LinearSystem(A, B)

    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()

    # Проверка, что левая матрица приведена к единичной форме
    np.testing.assert_array_equal(linear_system.A, np.eye(A.shape[0], A.shape[1]))

    # Проверка, что решение совпадает с ожидаемым
    np.testing.assert_array_equal(linear_system.B, X)
