from typing import Any, Tuple

import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt

from livy.dedup.svc import Service


class SIFTService(Service):
    _TRESHOLD = 0.3

    _sift: cv.SIFT

    def __init__(self) -> None:
        super().__init__()

        self._sift = cv.SIFT_create()

    def is_duplicate(self, orig_im: np.ndarray, dup_im: np.ndarray) -> bool:
        _, orig_desc = self._descriptor(orig_im)
        _, dup_desc = self._descriptor(dup_im)

        bf = cv.BFMatcher(cv.NORM_L2)

        potential_matches = bf.knnMatch(orig_desc, dup_desc, k=2)
        matches = []

        for (a, b) in potential_matches:
            if a.distance < 0.8 * b.distance:
                matches.append([a])

        # match_im = cv.drawMatchesKnn(orig_im, orig_kp, dup_im, dup_kp, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return self._TRESHOLD * len(potential_matches) <= len(matches)

    def _descriptor(self, im: np.ndarray) -> Tuple[Any, np.ndarray]:
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
   
        kp, des = self._sift.detectAndCompute(im_gray, None)

        # im = cv.drawKeypoints(im_gray, kp, im, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imwrite("sift_kpts.jpg", im)

        return kp, des
