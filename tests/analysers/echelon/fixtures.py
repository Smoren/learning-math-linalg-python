from dataclasses import dataclass
from typing import Generator, Set

import numpy as np

@dataclass
class AnalyseResult:
    is_square: bool
    is_reduced_echelon: bool
    is_zero: bool
    is_identity: bool
    rank: int
    pivot_columns: Set[int]


def data_provider_for_matrix_echelon_analyser() -> Generator[tuple[np.ndarray, AnalyseResult], None, None]:
    yield from [
        (
            np.array([
                [1, 2, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ], dtype=np.float64),
            AnalyseResult(
                is_square=True,
                is_reduced_echelon=True,
                is_zero=False,
                is_identity=False,
                rank=2,
                pivot_columns={0, 2},
            ),
        ),
    ]
