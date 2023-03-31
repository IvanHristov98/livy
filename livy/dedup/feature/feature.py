import livy.dedup.service as service

import numpy as np
import cv2 as cv

from typing import Tuple


class SIFTExtractor(service.Extractor):
    _sift: cv.SIFT

    def __init__(self) -> None:
        self._sift = cv.SIFT_create()

    def descriptor(self, im: np.ndarray) -> Tuple[np.ndarray]:
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
   
        _, des = self._sift.detectAndCompute(im_gray, None)

        # im = cv.drawKeypoints(im_gray, kp, im, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imwrite("sift_kpts.jpg", im)

        return des
