from typing import List

import livy.id as id
import livy.model as model
import livy.dedup.service as service

class SignatureService(service.Service):
    _extractor: service.Extractor

    def __init__(self, extractor: service.Extractor) -> None:
        self._extractor = extractor

    def add_im(self, im: model.Image) -> id.Image:
        pass

    def im(self, id: id.Image) -> model.Image:
        return self._store[id]

    def similar_ims(self, im: model.Image, n: int) -> List[id.Image]:
        pass
