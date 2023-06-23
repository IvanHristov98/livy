import abc
from typing import List, NamedTuple

import numpy as np
import cv2 as cv

import livy.id as id
import livy.model as model
import livy.dedup.service as service


class Image(NamedTuple):
    im: model.Image
    descriptor: np.ndarray


class ImageIterator(abc.ABC):
    def next(self) -> bool:
        raise NotImplementedError("next is not implemented")

    def curr(self) -> Image:
        raise NotImplementedError("curr is not implemented")


class ImageNotFound(Exception):
    """
    Thrown whenever an image is not found in the store.
    """


class ImageStore(abc.ABC):
    def add(self, im: Image) -> None:
        raise NotImplementedError("add is not implemented")

    def im(self, im_id: id.Image) -> model.Image:
        raise NotImplementedError("im is not implemented")

    def iterator(self) -> ImageIterator:
        raise NotImplementedError("iterator is not implemented")


class BruteForceService(service.Service):
    _neighbour_threshold: float
    _extractor: service.Extractor
    _store: ImageStore

    def __init__(
            self, 
            extractor: service.Extractor, 
            store: ImageStore, 
            neighbour_threshold: float = 0.8,
        ) -> None:
        self._neighbour_threshold = neighbour_threshold
        self._extractor = extractor
        self._store = store

    def add_im(self, im: model.Image) -> id.Image:
        descriptor = self._extractor.features(im.mat)

        self._store.add(Image(im, descriptor))
        return im.id

    def im(self, im_id: id.Image) -> model.Image:
        return self._store.im(im_id)

    def similar_ims(self, im: model.Image, n: int) -> List[id.Image]:
        descriptor = self._extractor.features(im.mat)
        
        class Score(NamedTuple):
            id: id.Image
            score: float

        top_scores: List[Score] = []

        for i in range(n+1):
            top_scores.append(Score(id=id.NoImage, score=0.0))

        store_iter = self._store.iterator()

        while store_iter.next():
            other_im = store_iter.curr()

            score = self._match_score(descriptor, other_im.descriptor)
            top_scores[n] = Score(id=other_im.im.id, score=score)

            # An insert sort is used to order small number of items fast.
            # Other sorts would be slow.
            for i in range(n-1, -1, -1):
                if top_scores[i].score > score:
                    break

                top_scores[i+1], top_scores[i] = top_scores[i], top_scores[i+1]
    
        top_ims: List[id.Image] = []

        for score in top_scores:
            top_ims.append(score.id)

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
