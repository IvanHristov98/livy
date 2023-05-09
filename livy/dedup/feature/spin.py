from typing import Tuple, List
import queue
import math

import cv2 as cv
import numpy as np

import livy.dedup.service as service


class SpinImageExtractor(service.Extractor):
    _spin_im_dims: int
    _sift: cv.SIFT

    def __init__(self, spin_im_dims: int = 20) -> None:
        self._spin_im_dims = spin_im_dims
        self._sift = cv.SIFT_create()

    # TODO: Experiment with other feature extractors.
    def features(self, im: np.ndarray) -> np.ndarray:
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        keypoints, _ = self._sift.detectAndCompute(im_gray, None)
        features = []

        for i in range(len(keypoints)):
            # pt is returned in (x, y) order
            im_pt = keypoints[i].pt
            im_radius = keypoints[i].size

            blob = self._fit_blob(im_gray, im_pt, im_radius)
            blob_h, _ = blob.shape
            blob_radius = math.floor(blob_h / 2)

            if blob_radius < 2:
                # Hard to get valuable info from point with no radius.
                continue

            spin_im = self._spin_im(blob, (blob_radius, blob_radius), blob_radius)
            features.append(spin_im.flatten())

            # cv.imshow("spin_im", (spin_im).astype(np.uint8))
            # cv.imshow("blob", blob)
            # cv.waitKey(delay=0)
            # if i > 10:
            #     break

        # im = cv.drawKeypoints(im_gray, keypoints, im, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return np.asarray(features, dtype=np.float32)

    # fit_blob resizes the blob so that feature extraction could work faster.
    def _fit_blob(self, im: np.ndarray, pt_fl: Tuple[float, float], radius_fl: float) -> np.ndarray:
        # Discretisize the blob.
        pt = (math.floor(pt_fl[0]), math.floor(pt_fl[1]))
        radius = math.ceil(radius_fl)
        boundary_pts = [
            (pt[0]-radius, pt[1]), (pt[0]+radius, pt[1]),
            (pt[0], pt[1]-radius), (pt[0], pt[1]+radius),
        ]

        while True:
            if self._are_in_bounds(im, boundary_pts):
                break
        
            radius -= 1

            boundary_pts[0] = (pt[0]-radius, pt[1])
            boundary_pts[1] = (pt[0]+radius, pt[1])
            boundary_pts[2] = (pt[0], pt[1]-radius)
            boundary_pts[3] = (pt[0], pt[1]+radius)

        # Resize the blob.
        _MAX_RADIUS = 5
        blob = im[pt[1]-radius:pt[1]+radius+1, pt[0]-radius:pt[0]+radius+1]

        if radius < _MAX_RADIUS:
            return blob

        return cv.resize(blob, (_MAX_RADIUS*2+1, _MAX_RADIUS*2+1))

    def _are_in_bounds(self, im, pts: List[Tuple[int, int]]) -> bool:
        for pt in pts:
            if not self._is_in_bounds(im, pt):
                return False
        return True

    # _spin_im returns the spin image of a blob on a single channel image.
    def _spin_im(self, im: np.ndarray, center: Tuple[float, float], radius: float) -> np.ndarray:
        spin_im = np.zeros((self._spin_im_dims, self._spin_im_dims), dtype=np.float32)
        
        fringe = queue.Queue()
        fringe.put(center)

        visited = set([(math.floor(center[0]), math.floor(center[1]))])
        neighbours = [0] * 8

        while not fringe.empty():
            pt = fringe.get()

            norm_intensity = float(im[math.floor(pt[0]), math.floor(pt[1])]) / 255.0
            norm_dist = self._dist(center, pt) / radius

            spin_im = self._fill_spin_im(spin_im, norm_intensity, norm_dist)

            # 8-neighbor relationship.
            neighbours[0] = (pt[0]-1, pt[1])
            neighbours[1] = (pt[0]+1, pt[1])
            neighbours[2] = (pt[0], pt[1]-1)
            neighbours[3] = (pt[0], pt[1]+1)
            neighbours[4] = (pt[0]-1, pt[1]-1)
            neighbours[5] = (pt[0]-1, pt[1]+1)
            neighbours[6] = (pt[0]+1, pt[1]-1)
            neighbours[7] = (pt[0]+1, pt[1]+1)

            # Add unvisited neighbours to fringe.
            for neighbour in neighbours:
                if not self._is_in_bounds(im, neighbour):
                    continue

                neighbour_pix = (math.floor(neighbour[0]), math.floor(neighbour[1]))

                if neighbour_pix in visited or self._dist(center, neighbour) > radius:
                    continue

                fringe.put(neighbour)
                visited.add(neighbour_pix)

        return spin_im

    def _dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

    def _is_in_bounds(self, im: np.ndarray, pt: Tuple[float, float]) -> bool:
        h, w = im.shape

        return not(pt[0] < 0 or pt[0] > w-1 or pt[1] < 0 or pt[1] > h-1)

    # _fill_spin_im takes a spin image, fills it up and returns a reference to the modified spin_im.
    def _fill_spin_im(
            self, 
            spin_im: np.ndarray,
            # norm_intensity is from 0 to 1. Refers to i in paper.
            norm_intensity: float,
            # norm_dist is from 0 to 1. Refers to d in paper.
            norm_dist: float,
        ) -> np.ndarray:
        _MIN_SCORE_THRESHOLD = 0.85

        # Index logic.
        #
        # spin_h = number of intensity buckets.
        # spin_w = number of distance buckets.
        spin_h, spin_w = spin_im.shape

        intensity_step = 1.0/(spin_h-1)
        intensity_idx = math.floor(norm_intensity / intensity_step)

        dist_step = 1.0/(spin_w-1)
        dist_idx = math.floor(norm_dist / dist_step)

        # BFS logic.
        fringe = queue.Queue()
        fringe.put((intensity_idx, dist_idx))

        visited = set([(intensity_idx, dist_idx)])
        cnt = 0

        while not fringe.empty():
            pt = fringe.get()
            cnt += 1

            # Refers to i0 in paper.
            cell_intensity = float(intensity_step * pt[0])
            # Refers to d0 in paper.
            cell_dist = float(dist_step * pt[1])

            dist_contr = math.sqrt(abs((cell_dist-norm_dist))) / 2
            intensity_contr = math.sqrt(abs((cell_intensity-norm_intensity))) / 2
            score = math.exp(-(dist_contr + intensity_contr))

            # print(dist_contr + intensity_contr, score)
            spin_im[pt[0], pt[1]] += score

            if score <  _MIN_SCORE_THRESHOLD:
                continue

            # 4-neighbor relationship.
            # Affects only performance.
            neighbours = [(pt[0]-1, pt[1]), (pt[0]+1, pt[1]), (pt[0], pt[1]-1), (pt[0], pt[1]+1)]

            for neighbour in neighbours:
                if neighbour in visited:
                    continue
                
                # Dimensions are reversed because in images ys and xs are reversed.
                if not self._is_in_bounds(spin_im, (float(neighbour[1]), float(neighbour[0]))):
                    continue

                fringe.put(neighbour)
                visited.add(neighbour)

        return spin_im
