import abc

import numpy as np
import cv2 as cv

from typing import Tuple


class Extractor(abc.ABC):
    def descriptor(self, im: np.ndarray) -> Tuple[np.ndarray]:
        raise NotImplementedError("descriptor is not implemented")


class DuplicateChecker(abc.ABC):
    def is_duplicate(self, base_desc: np.ndarray, dup_desc: np.ndarray) -> bool:
        raise NotImplementedError("descriptor is not implemented")


class Service:
    _TRESHOLD = 0.3

    _extractor: Extractor
    _checker: DuplicateChecker

    def __init__(self, extractor: Extractor, checker: DuplicateChecker) -> None:
        self._extractor = extractor
        self._checker = checker

    def is_duplicate(self, base_im: np.ndarray, dup_im: np.ndarray) -> bool:
        base_desc = self._extractor.descriptor(base_im)
        dup_desc = self._extractor.descriptor(dup_im)

        return self._checker.is_duplicate(base_desc, dup_desc)

