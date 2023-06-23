from typing import Dict, List

import livy.dedup.service.siftknn as siftknn
import livy.id as id


class ImageStore(siftknn.ImageStore):
    _ims: Dict[id.Image, siftknn.Image]

    def __init__(self) -> None:
        super().__init__()
        self._ims = dict()

    def add(self, im: siftknn.Image) -> None:
        self._ims[im.im.id] = im

    def im(self, im_id: id.Image) -> siftknn.Image:
        return self._ims[im_id]

    def iterator(self) -> siftknn.ImageIterator:
        return ImageIterator(self)


class ImageIterator(siftknn.ImageIterator):
    _ims: List[siftknn.Image]
    _curr_idx: int

    def __init__(self, store: ImageStore) -> None:
        super().__init__()
        self._ims = []
        for im in store._ims.values():
            self._ims.append(im)

        self._curr_idx = -1

    def next(self) -> bool:
        if self._curr_idx+1 == len(self._ims):
            return False
        
        self._curr_idx += 1
        return True

    def curr(self) -> siftknn.Image:
        return self._ims[self._curr_idx]
