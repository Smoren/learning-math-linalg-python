import numpy as np
import pytest

from app.solvers import get_determinant
from tests.solvers.derterminant.fixtures import data_provider_for_determinant


@pytest.mark.parametrize("data", data_provider_for_determinant())
def test_get_inverse_matrix(data: tuple[np.ndarray]):
    A, = data

    d_actual = get_determinant(A)
    d_expected = np.linalg.det(A)

    np.testing.assert_almost_equal(d_actual, d_expected)
