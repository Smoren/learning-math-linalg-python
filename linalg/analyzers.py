import numpy as np

from linalg.system import LinearSystem
from linalg.utils import is_zero


class LinearSystemBaseAnalyser:
    _linear_system: LinearSystem

    def __init__(self, linear_system: LinearSystem):
        self._linear_system = linear_system

    def is_homogeneous(self) -> bool:
        return np.all(is_zero(self._linear_system.B))

    def is_square(self) -> bool:
        return self._linear_system.A.shape[0] == self._linear_system.A.shape[1]

    def is_singular(self) -> bool:
        return not self.is_square() or is_zero(np.linalg.det(self._linear_system.A))
