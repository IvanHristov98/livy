import abc
from typing import List

import numpy as np

import livy.id as id
import livy.model as model


class Extractor(abc.ABC):
    # features returns an array of k 1xn-dimensional features for an image.
    def features(self, im: np.ndarray) -> np.ndarray:
        raise NotImplementedError("descriptor is not implemented")


class Service(abc.ABC):
    def add_im(self, im: model.Image) -> id.Image:
        raise NotImplementedError("add_im is not implemented")

    def im(self, id: id.Image) -> model.Image:
        raise NotImplementedError("im is not yet implemented")

    def similar_ims(self, im: model.Image, n: int) -> List[id.Image]:
        raise NotImplementedError("similar_ims is not implemented")

