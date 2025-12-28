import numpy as np

from linalg.system import LinearSystem


class LinearSystemBaseAnalyser:
    _linear_system: LinearSystem

    def __init__(self, linear_system: LinearSystem):
        self._linear_system = linear_system

    def is_homogeneous(self) -> bool:
        return np.all(self._is_zero(self._linear_system.B))

    def _is_zero(self, x):
        return np.abs(x) <= np.finfo(self._linear_system.A.dtype).eps
