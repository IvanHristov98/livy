import livy.id as id

import numpy as np

from typing import NamedTuple


class Image(NamedTuple):
    id: id.Image
    name: str
    mat: np.ndarray
