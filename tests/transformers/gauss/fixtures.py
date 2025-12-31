from typing import Generator

import numpy as np


def data_provider_for_gauss() -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    yield from [
        (
            np.array([
                [0, 2, 3, 4],
                [4, 0, 5, 1],
                [5, 6, 0, 8],
                [6, 5, 1, 0],
            ], dtype=np.float64),
            np.array([
                [1],
                [2],
                [3],
                [4],
            ], dtype=np.float64),
            np.array([
                [0.2736196319018405],
                [0.4306748466257669],
                [0.2049079754601227],
                [-0.11901840490797544],
            ], dtype=np.float64),
        ),
        (
            np.array([
                [0, 2, 0, 0],
                [4, 0, 0, 0],
                [0, 0, 0, 8],
                [0, 0, 1, 0],
            ], dtype=np.float64),
            np.array([
                [1],
                [2],
                [3],
                [4],
            ], dtype=np.float64),
            np.array([
                [0.5],
                [0.5],
                [4.0],
                [0.375],
            ], dtype=np.float64),
        ),
    ]
