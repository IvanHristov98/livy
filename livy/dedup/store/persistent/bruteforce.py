from typing import List
from pathlib import Path
import os
import uuid

import cv2 as cv
import numpy as np

import livy.dedup.service.bruteforce as bruteforce
import livy.id as id


class ImageStore(bruteforce.ImageStore):
    _vol: Path

    def __init__(self, vol: Path) -> None:
        super().__init__()
        self._vol = vol

        _im_volume(vol).mkdir(parents=True, exist_ok=True)
        _descriptor_vol(vol).mkdir(parents=True, exist_ok=True)

    def add(self, im: bruteforce.Image) -> None:
        cv.imwrite(_im_path(self._vol, im.im.id), im.im.mat)
        im.descriptor.tofile(_descriptor_path(self._vol, im.im.id))

    def im(self, im_id: id.Image) -> bruteforce.Image:
        return _read_im(self._vol, im_id)

    def iterator(self) -> bruteforce.ImageIterator:
        return ImageIterator(self)


class ImageIterator(bruteforce.ImageIterator):
    _vol: Path
    _im_ids: List[id.Image]
    _curr_idx: int

    def __init__(self, store: ImageStore) -> None:
        super().__init__()

        self._vol = store._vol
        # TODO: Optimise usage of im_ids. Listing the whole directory is slow.
        self._im_ids = []

        ims_vol = _im_volume(store._vol)
        for im_name in os.listdir(str(ims_vol)):
            if os.path.isfile(os.path.join(str(ims_vol), im_name)):
                raw_im_id = im_name.split(".")[0]
                im_id = uuid.UUID(raw_im_id)

                self._im_ids.append(im_id)                

        self._curr_idx = -1

    def next(self) -> bool:
        if self._curr_idx+1 == len(self._im_ids):
            return False

        self._curr_idx += 1
        return True

    def curr(self) -> bruteforce.Image:
        return _read_im(self._vol, self._im_ids[self._curr_idx])


def _read_im(vol: Path, im_id: id.Image) -> bruteforce.Image:
    if not os.path.exists(_im_path(vol, im_id)):
        raise bruteforce.ImageNotFound(f"failed finding image for {str(im_id)}")

    mat = cv.imread(_im_path(vol, im_id))
    descriptor = np.fromfile(_descriptor_path(vol, im_id))

    return bruteforce.Image(im=mat, descriptor=descriptor)


def _im_path(vol: Path, im_id: id.Image) -> str:
    return os.path.join(str(_im_volume(vol)), f"{str(im_id)}.jpg")


def _im_volume(vol: Path) -> Path:
    return Path(vol, "ims")


def _descriptor_path(vol: Path, im_id: id.Image) -> str:
    return os.path.join(str(_descriptor_vol(vol)), f"{str(im_id)}.descriptor")


def _descriptor_vol(vol: Path) -> Path:
    return Path(vol, "descriptors")
