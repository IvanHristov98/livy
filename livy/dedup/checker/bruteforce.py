import livy.dedup.service as service

import numpy as np
import cv2 as cv


class BruteForceChecker(service.DuplicateChecker):
    _NEIGHBOUR_THRESH = 0.8
    _MIN_MATCH_RATIO = 0.5

    def is_duplicate(self, base_desc: np.ndarray, dup_desc: np.ndarray) -> bool:
        bf = cv.BFMatcher(cv.NORM_L2)

        potential_matches = bf.knnMatch(base_desc, dup_desc, k=2)
        matches = []

        for (a, b) in potential_matches:
            if a.distance < self._NEIGHBOUR_THRESH * b.distance:
                matches.append([a])

        # match_im = cv.drawMatchesKnn(orig_im, orig_kp, dup_im, dup_kp, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # plt.imshow(match_im)
        # plt.waitforbuttonpress()

        return self._MIN_MATCH_RATIO * len(potential_matches) <= len(matches)
