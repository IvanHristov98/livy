from typing import List, Dict, NamedTuple

import numpy as np
import cv2 as cv

import livy.id as id
import livy.model as model
import livy.dedup.service as service

class Image(NamedTuple):
    im: model.Image
    descriptor: np.ndarray


class BruteForceService(service.Service):
    _neighbour_threshold: float
    _extractor: service.Extractor
    _store: Dict[id.Image, Image]

    def __init__(self, extractor: service.Extractor, neighbour_threshold: float = 0.8) -> None:
        self._neighbour_threshold = neighbour_threshold
        self._extractor = extractor
        self._store = dict()

    def add_im(self, im: model.Image) -> id.Image:
        _, descriptor = self._extractor.features(im.mat)

        self._store[im.id] = Image(im, descriptor)
        return im.id

    def im(self, id: id.Image) -> model.Image:
        return self._store[id]

    def similar_ims(self, im: model.Image, n: int) -> List[id.Image]:
        descriptor = self._extractor.features(im.mat)
        
        class Score(NamedTuple):
            id: id.Image
            score: float

        top_ims: List[Score] = []

        for i in range(n+1):
            top_ims.append(Score(id=id.NoImage, score=0.0))

        for _, other_im in self._store.items():

            score = self._match_score(descriptor, other_im.descriptor)
            top_ims[n] = Score(id=other_im.im.id, score=score)

            # An insert sort is used to order small number of items fast.
            # Other sorts would be slow.
            for i in range(n-1, -1, -1):
                if top_ims[i].score > score:
                    break

                top_ims[i+1], top_ims[i] = top_ims[i], top_ims[i+1]

        return top_ims[:n]

    def _match_score(self, fst_desc: np.ndarray, snd_desc: np.ndarray) -> float:
        bf = cv.BFMatcher(cv.NORM_L2)

        potential_matches = bf.knnMatch(fst_desc, snd_desc, k=2)
        matches = []

        for (a, b) in potential_matches:
            if a.distance < self._neighbour_threshold * b.distance:
                matches.append([a])

        # match_im = cv.drawMatchesKnn(orig_im, orig_kp, dup_im, dup_kp, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # plt.imshow(match_im)
        # plt.waitforbuttonpress()

        return len(matches) / len(potential_matches)
