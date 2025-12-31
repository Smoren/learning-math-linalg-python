import numpy as np
import pytest

from app.operations import add_matrices, mul_matrices
from tests.operations.base.fixtures import data_provider_for_add_matrices, data_provider_for_mul_matrices


@pytest.mark.parametrize("data", data_provider_for_add_matrices())
def test_add_matrices(data: tuple[np.ndarray, np.ndarray]):
    A, B = data

    S_actual = add_matrices(A, B)
    S_expected = A + B

    np.testing.assert_array_equal(S_actual, S_expected)


@pytest.mark.parametrize("data", data_provider_for_mul_matrices())
def test_mul_matrices(data: tuple[np.ndarray, np.ndarray]):
    A, B = data

    S_actual = mul_matrices(A, B)
    S_expected = A @ B

    np.testing.assert_array_equal(S_actual, S_expected)
