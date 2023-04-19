import numpy as np
import livy.id as id

from typing import NamedTuple


class Image(NamedTuple):
    id: id.Image
    name: str
    mat: np.ndarray
