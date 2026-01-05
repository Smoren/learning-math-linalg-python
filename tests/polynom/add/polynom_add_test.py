import numpy as np
import pytest

from app.polynom import MatrixPolynom
from tests.polynom.add.fixtures import data_provider_for_add_polynomials


@pytest.mark.parametrize("data", data_provider_for_add_polynomials())
def test_add_polynomials(data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    lhs_coefficients, rhs_coefficients, expected_coefficients = data

    lhs = MatrixPolynom(lhs_coefficients)
    rhs = MatrixPolynom(rhs_coefficients)
    result = lhs + rhs

    np.testing.assert_array_equal(result.coefficients, expected_coefficients)
