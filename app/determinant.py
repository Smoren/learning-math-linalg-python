import numpy as np


def get_determinant(matrix: np.ndarray) -> float:
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

    if matrix.shape[0] == 1:
        return matrix[0][0]

    result = 0

    for i in range(matrix.shape[1]):
        m = get_minor(matrix, 0, i)
        a = matrix[0][i]
        result += (-1)**i * a * get_determinant(m)

    return result


def get_minor(matrix: np.ndarray, i: int, j: int) -> np.ndarray:
    top_left, top_right = matrix[:i, :j], matrix[:i, j+1:]
    bottom_left, bottom_right = matrix[i+1:, :j], matrix[i+1:, j+1:]

    top = np.concatenate((top_left, top_right), axis=1)
    bottom = np.concatenate((bottom_left, bottom_right), axis=1)

    return np.concatenate((top, bottom), axis=0)
