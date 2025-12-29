import numpy as np


def create_transform_matrix_add_row(index_from: int, index_to: int, n: int, mult: float) -> np.ndarray:
    """
    Создает матрицу элементарного преобразования n×n, прибавляющую к строке с индексом index_to
    строку с индексом index_from, умноженную на mult.
    """
    # Проверяем, что индексы находятся в допустимом диапазоне
    assert 0 <= index_from < n and 0 <= index_to < n

    # Создаем единичную матрицу и добавляем множитель в ячейку (index_to, index_from)
    transformation_matrix = np.eye(n)
    transformation_matrix[index_to, index_from] = mult

    return transformation_matrix


def create_transform_matrix_mul_row(index: int, mult: float, n: int) -> np.ndarray:
    """
    Создает матрицу элементарного преобразования n×n, умножающую строку с индексом index на mult.
    """
    # Проверяем, что индекс находится в допустимом диапазоне
    assert 0 <= index < n

    # Создаем единичную матрицу и добавляем множитель в ячейку (index, index)
    transformation_matrix = np.eye(n)
    transformation_matrix[index, index] = mult

    return transformation_matrix


def create_transform_matrix_swap_rows(lhs_index: int, rhs_index: int, n: int) -> np.ndarray:
    """
    Создает матрицу элементарного преобразования n×n, меняющую местами строки с индексами lhs_index и rhs_index.
    """
    # Проверяем, что индексы находятся в допустимом диапазоне
    assert 0 <= lhs_index < n and 0 <= rhs_index < n

    # Создаем единичную матрицу
    transformation_matrix = np.eye(n)

    # Устанавливаем единицы в ячейки (lhs_index, rhs_index) и (rhs_index, lhs_index)
    transformation_matrix[lhs_index, rhs_index] = 1
    transformation_matrix[rhs_index, lhs_index] = 1

    # Обнуляем диагональные ячейки (lhs_index, lhs_index) и (rhs_index, rhs_index), "компенсируя" добавленные единицы в этих столбцах
    transformation_matrix[lhs_index, lhs_index] = 0
    transformation_matrix[rhs_index, rhs_index] = 0

    return transformation_matrix
