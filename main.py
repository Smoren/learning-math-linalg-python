import numpy as np

from linalg.determinant import get_minor, get_determinant

if __name__ == '__main__':
    a = np.array([
        [1, 2, 3, 4],
        [4, 3, 5, 1],
        [5, 6, 7, 8],
        [6, 5, 1, 2],
    ])

    m = get_determinant(a)
    print(m)
