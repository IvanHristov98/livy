import unittest
from typing import List, Tuple
from pathlib import Path
import os
import os.path
import random
import math
import time

import numpy as np
import cv2 as cv

import livy.dedup as dedup
import livy.model as model
import livy.id as id


class TestDedupService(unittest.TestCase):
    _ims_path: Path
    _svc: dedup.Service

    def setUp(self) -> None:
        super().setUp()
        random.seed(int(time.time()))

        self._ims_path = Path(os.environ["IMS_PATH"])

        # TODO: Make many services.
        # self._svc = self._new_brute_force_svc()
        self._svc = self._new_signature_svc()

    def _new_brute_force_svc(self) -> dedup.BruteForceService:
        extractor = dedup.SIFTExtractor()
        return dedup.BruteForceService(extractor)

    def _new_signature_svc(self) -> dedup.SignatureService:
        extractor = dedup.SpinImageExtractor()
        return dedup.SignatureService(extractor)

    def test_affine_transform_stability(self) -> None:
        ims = self._load_ims()[:20]
       
        for i in range(len(ims)):
            self._svc.add_im(ims[i])
        
        print("loaded images")

        sample_size = 5
        success_count = 0

        warped_ims = self._warp_affine_ims(ims)
        
        for i in range(sample_size):
            idx = random.randrange(0, len(warped_ims))
            similar_ims = self._svc.similar_ims(warped_ims[idx], n=5)

            for similar_im in similar_ims:
                if similar_im.id == ims[idx].id:
                    success_count += 1
                    break

        print("top n success rate: ", success_count/sample_size)

    def _warp_affine_ims(
        self, 
        ims: List[model.Image], 
        max_angle: float = 45.0, 
        max_offset_ratio: float = 0.1, 
        max_scale_offset: float = 0.5,
    ) -> List[model.Image]:
        warped_ims = [None] * len(ims)

        for i in range(0, len(ims)):
            (h, w) = ims[i].mat.shape[:2]

            angle = random.uniform(-max_angle, max_angle)

            offx = w * random.uniform(-max_offset_ratio, max_offset_ratio)
            offy = h * random.uniform(-max_offset_ratio, max_offset_ratio)

            scalex = 1 + random.uniform(-max_scale_offset, max_scale_offset)
            scaley = 1 + random.uniform(-max_scale_offset, max_scale_offset)

            mat = self._affine_transform_mat(ims[i].mat, angle, (offx, offy), (scalex, scaley))
            mat = cv.warpAffine(ims[i].mat, mat, (w, h))

            warped_ims[i] = model.Image(id.NewImage(), f"test-{str(i)}", mat)

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
        in_pts = np.float32([[float(0), float(0)], [float(w-1), float(0)], [float(0), float(h-1)]])
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

    def _load_ims(self) -> List[model.Image]:
        paths = self._im_paths()
        ims = [None] * len(paths)

        for i in range(0, len(paths)):
            mat = cv.imread(str(paths[i]))
            ims[i] = model.Image(id.NewImage(), str(i), mat)

        return ims

    def _im_paths(self) -> List[Path]:
        paths = []
        
        for name in os.listdir(str(self._ims_path)):
            if len(paths) > 100:
                break

            path = Path(self._ims_path, name)
            if os.path.isfile(path):
                paths.append(Path(path))
        
        return paths
