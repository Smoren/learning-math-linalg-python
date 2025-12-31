import numpy as np
import pytest

from app.system import LinearSystem
from app.transformers import LinearSystemGaussTransformer
from tests.transformers.gauss.fixtures import data_provider_for_gauss


@pytest.mark.parametrize("data", data_provider_for_gauss())
def test_gauss_for_invertible(data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    A, B, B_expected = data

    linear_system = LinearSystem(A, B)

    transformer = LinearSystemGaussTransformer(linear_system)
    transformer.apply_gauss()

    assert np.isclose(linear_system.A, np.eye(A.shape[0], A.shape[1])).all()
    assert np.isclose(linear_system.B, B_expected).all()
