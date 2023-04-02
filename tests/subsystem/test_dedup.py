import unittest
from typing import List, Tuple
from pathlib import Path
import os
import os.path
import random

import numpy as np
import cv2 as cv

import livy.dedup as dedup

class TestDedupService(unittest.TestCase):
    _MAX_ANGLE = 90.0

    _ims_path: Path
    _svc: dedup.Service

    def setUp(self) -> None:
        super().setUp()
        self._ims_path = os.environ["COCO_IMS_PATH"]

        extractor = dedup.SIFTExtractor()
        dup_checker = dedup.BruteForceChecker()

        self._svc = dedup.Service(extractor, dup_checker)

    def test_affine_transform_stability(self) -> None:
        ims = self._load_ims()
        tuples = self._rotate_ims(ims)
        success_count = 0

        for (im, warp_im) in tuples:
            if self._svc.is_duplicate(im, warp_im):
                success_count += 1

        print("success count: ", success_count/len(ims))

    def _scale_ims(self, ims: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        pass

    def _translate_ims(self, ims: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        pass

    def _rotate_ims(self, ims: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        tuples = [None] * len(ims)

        for i in range(0, len(ims)):
            (h, w) = ims[i].shape[:2]
            (cY, cX) = (h//2, w//2)
            angle = random.uniform(-self._MAX_ANGLE, self._MAX_ANGLE)

            rot_mat = cv.getRotationMatrix2D((cX, cY), angle,1.0)
            rot_im = cv.warpAffine(ims[i], rot_mat, (w, h))

            tuples[i] = (ims[i], rot_im)
        
        return tuples

    def _load_ims(self) -> List[np.ndarray]:
        paths = self._im_paths()
        ims = [None] * len(paths)

        for i in range(0, len(paths)):
            ims[i] = cv.imread(str(paths[i]))

        return ims

    def _im_paths(self) -> List[Path]:
        paths = []
        
        for name in os.listdir(self._ims_path):
            if len(paths) > 100:
                break

            path = Path(self._ims_path, name)
            if os.path.isfile(path):
                paths.append(Path(path))
        
        return paths
