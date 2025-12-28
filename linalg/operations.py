import numpy as np


def add_matrices(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Сложение двух матриц"""
    # Проверяем, что матрицы имеют одинаковые размеры
    assert lhs.shape == rhs.shape

    # Резльтирующая матрица будет иметь те же размеры, что и любое из слагаемых
    result = np.zeros_like(lhs)

    # Складываем соответствующие элементы матриц
    for i in range(lhs.shape[0]):
        for j in range(lhs.shape[1]):
            result[i, j] = lhs[i, j] + rhs[i, j]

    return result


def mul_matrix(matrix, multiplier: float) -> np.ndarray:
    """Умножение матрицы на число"""
    # Результирующая матрица будет иметь те же размеры, что и исходная
    result = np.zeros_like(matrix)

    # Умножаем каждый элемент матрицы на число
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result[i, j] = matrix[i, j] * multiplier

    return result

def mul_matrices(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Умножение двух матриц"""
    # Количество столбцов левой матрицы должно равняться количеству строк правой матрицы
    assert lhs.shape[1] == rhs.shape[0]

    # Новая матрица будет иметь размеры (количество строк левого, количество столбцов правого)
    result = np.zeros((lhs.shape[0], rhs.shape[1]))

    # Проходим по каждой ячейке новой матрицы
    for i in range(result.shape[0]):
        # Проходим по каждой ячейке строки i в новой матрице
        for j in range(result.shape[1]):
            # Считаем значение ячейки i, j в новой матрице
            cell_value = 0
            # Проходим по строке в левой матрице и столбцу в правой матрице
            for k in range(lhs.shape[1]):
                # и суммируем произведения
                cell_value += lhs[i, k] * rhs[k, j]
            # Присваиваем значение ячейки i, j в новой матрице
            result[i, j] = cell_value

    return result
