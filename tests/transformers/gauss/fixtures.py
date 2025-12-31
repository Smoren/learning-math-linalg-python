from typing import Generator

import numpy as np


def data_provider_for_gauss_for_invertible() -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
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
        ),
        (
            np.array([
                [1, 2, 3, 0],
                [2, 4, 5, 0],
                [0, 0, 0, 1],
                [1, -4, 7, 0],
            ], dtype=np.float64),
            np.array([
                [1],
                [2],
                [3],
                [4],
            ], dtype=np.float64),
        ),
    ]


def data_provider_for_gauss_for_singular() -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    yield from [
        (
            np.array([
                [0, 2, 0, 0],
                [4, 0, 1, 0],
                [0, 0, 0, 8],
                [4, 0, 1, 0],
            ], dtype=np.float64),
            np.array([
                [1],
                [2],
                [3],
                [4],
            ], dtype=np.float64),
        ),
        (
            np.array([
                [1, 0, 3, 0],
                [2, 0, 5, 0],
                [0, 0, 0, 1],
                [1, 0, 7, 0],
            ], dtype=np.float64),
            np.array([
                [1],
                [2],
                [3],
                [4],
            ], dtype=np.float64),
        ),
        (
            np.array([
                [0, 2, 3],
                [4, 0, 5],
                [5, 6, 0],
                [6, 5, 1],
            ], dtype=np.float64),
            np.array([
                [1],
                [2],
                [3],
                [4],
            ], dtype=np.float64),
        ),
        (
            np.array([
                [0, 2, 3, 4, 10],
                [4, 0, 5, 1, 11],
                [5, 6, 0, 8, 12],
                [6, 5, 1, 0, 13],
            ], dtype=np.float64),
            np.array([
                [1],
                [2],
                [3],
                [4],
            ], dtype=np.float64),
        ),
    ]
