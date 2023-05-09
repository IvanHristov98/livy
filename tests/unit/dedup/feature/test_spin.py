import unittest
from pathlib import Path
import os
import uuid

import cv2 as cv
import numpy as np

import livy.dedup as dedup


class TestSpinImageExtractor(unittest.TestCase):
    _extractor: dedup.SpinImageExtractor

    def setUp(self) -> None:
        self._extractor = dedup.SpinImageExtractor()

    def test_normalisation(self) -> None:
        im = self._sample_im()
        features = self._extractor.features(im)

        feature_cnt, dim_cnt = features.shape
        self.assertGreaterEqual(feature_cnt, 1)
        self.assertEqual(dim_cnt, 20*20)

    def _sample_im(self) -> np.ndarray:
        return cv.imread(str(self._sample_im_path()))

    def _sample_im_path(self) -> Path:
        ims_path = os.environ["IMS_PATH"]

        for name in os.listdir(ims_path):
            path = Path(Path(ims_path), name)
            if os.path.isfile(path):
                return path
        
        raise Exception("no image found in images path")
