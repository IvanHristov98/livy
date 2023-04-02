import unittest
from typing import List, Tuple
from pathlib import Path
import os
import os.path
import random
import math

import numpy as np
import cv2 as cv

import livy.dedup as dedup


class TestDedupService(unittest.TestCase):
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
        warped_ims = self._warp_affine_ims(ims)
        success_count = 0

        for i in range(len(ims)):
            if self._svc.is_duplicate(ims[i], warped_ims[i]):
                success_count += 1

        print("success count: ", success_count/len(ims))

    def _warp_affine_ims(
        self, 
        ims: List[np.ndarray], 
        max_angle: float = 45.0, 
        max_offset_ratio: float = 0.1, 
        max_scale_offset: float = 0.5,
    ) -> List[np.ndarray]:
        warped_ims = [None] * len(ims)

        for i in range(0, len(ims)):
            (h, w) = ims[i].shape[:2]

            angle = random.uniform(-max_angle, max_angle)

            offx = w * random.uniform(-max_offset_ratio, max_offset_ratio)
            offy = h * random.uniform(-max_offset_ratio, max_offset_ratio)

            scalex = 1 + random.uniform(-max_scale_offset, max_scale_offset)
            scaley = 1 + random.uniform(-max_scale_offset, max_scale_offset)

            mat = self._affine_transform_mat(ims[i], angle, (offx, offy), (scalex, scaley))
            warped_ims[i] = cv.warpAffine(ims[i], mat, (w, h))

            # if i < 10:
            #     # cv.imshow("image", ims[i])
            #     # cv.waitKey(0)
            #     cv.imshow("warped image", warped_ims[i])
            #     cv.waitKey(0)
        
        return warped_ims

    def _affine_transform_mat(
        self, 
        im: np.ndarray, 
        angle: float, 
        offsets: Tuple[float, float], 
        scales: Tuple[float, float],
    ) -> np.ndarray:
        (h, w) = im.shape[:2]
        in_pts = np.float32([[0, 0], [w-1, 0], [0, h-1]])
        out_pts = np.zeros_like(in_pts)

        for i in range(0, len(in_pts)):
            (cx, cy) = (w/2, h/2)
            (ix, iy) = in_pts[i]
            # Rotate.
            ox = math.cos(angle) * (ix - cx) - math.sin(angle) * (iy - cy) + cx
            oy = math.sin(angle) * (ix - cx) + math.cos(angle) * (iy - cy) + cy

            # Offset.
            ox += offsets[0]
            oy += offsets[1]
            
            # Scale.
            ox *= scales[0]
            oy *= scales[1]

            out_pts[i][0] = ox
            out_pts[i][1] = oy
        
        return cv.getAffineTransform(in_pts, out_pts)

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
