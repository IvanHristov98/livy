from typing import Dict, List

import livy.dedup.service.bruteforce as bruteforce
import livy.id as id


class ImageStore(bruteforce.ImageStore):
    _ims: Dict[id.Image, bruteforce.Image]

    def __init__(self) -> None:
        super().__init__()
        self._ims = dict()

    def add(self, im: bruteforce.Image) -> None:
        self._ims[im.im.id] = im

    def im(self, im_id: id.Image) -> bruteforce.Image:
        return self._ims[im_id]

    def iterator(self) -> bruteforce.ImageIterator:
        return ImageIterator(self)


class ImageIterator(bruteforce.ImageIterator):
    _ims: List[bruteforce.Image]
    _curr_idx: int

    def __init__(self, store: ImageStore) -> None:
        super().__init__()
        self._ims = store._ims.values()
        self._curr_idx = -1

    def next(self) -> bool:
        if self._curr_idx+1 == len(self._ims):
            return False
        
        self._curr_idx += 1
        return True

    def curr(self) -> bruteforce.Image:
        return self._ims[self._curr_idx]
