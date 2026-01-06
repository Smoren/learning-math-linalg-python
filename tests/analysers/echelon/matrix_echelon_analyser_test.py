import numpy as np
import pytest

from app.analysers import EchelonMatrixAnalyser
from tests.analysers.echelon.fixtures import AnalyseResult, data_provider_for_matrix_echelon_analyser


@pytest.mark.parametrize("data", data_provider_for_matrix_echelon_analyser())
def test_matrix_echelon_analyser(data: tuple[np.ndarray, AnalyseResult]):
    matrix, expected = data

    analyser = EchelonMatrixAnalyser(matrix)

    assert analyser.is_square() == expected.is_square
    assert analyser.is_reduced_echelon() == expected.is_reduced_echelon
    assert analyser.is_zero() == expected.is_zero
    assert analyser.is_identity() == expected.is_identity
    assert analyser.get_rank() == expected.rank
    assert analyser.get_pivot_columns() == expected.pivot_columns
