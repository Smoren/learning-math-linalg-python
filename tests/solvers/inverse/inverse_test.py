import numpy as np
import pytest

from app.solvers import get_inverse_matrix
from tests.solvers.inverse.fixtures import data_provider_for_inverse_matrix


@pytest.mark.parametrize("data", data_provider_for_inverse_matrix())
def test_get_inverse_matrix(data: tuple[np.ndarray]):
    A, = data
    Ai = get_inverse_matrix(A)

    np.testing.assert_array_almost_equal(A @ Ai, np.eye(A.shape[0]))
    np.testing.assert_array_almost_equal(Ai @ A, np.eye(A.shape[0]))
