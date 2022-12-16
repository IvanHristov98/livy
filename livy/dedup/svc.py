import abc

import numpy as np


class Service(abc.ABC):
    def is_duplicate(self, orig_im: np.ndarray, dup_im: np.ndarray) -> bool:
        raise NotImplementedError("is_duplicate is not yet implemented")
