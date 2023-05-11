from typing import List, NamedTuple, Dict
import math

import hdbscan
import numpy as np
import numpy.linalg as linalg

import livy.id as id
import livy.model as model
import livy.dedup.service as service


class WeightedMedoid(NamedTuple):
    # medoid is the most centrally located element of a cluster.
    medoid: np.ndarray
    # weight is between 0 and 1.
    weight: float

DiscreteHistogram = List[int]


class SignatureService(service.Service):
    # Hyper-parameters.
    _min_cluster_size: int
    _credits: int

    _extractor: service.Extractor
    _discrete_signatures: Dict[id.Image, DiscreteHistogram]
    _ims: Dict[id.Image, model.Image]

    def __init__(self, extractor: service.Extractor, min_cluster_size: int = 5, credits: int = 1000000) -> None:
        self._extractor = extractor
        self._discrete_signatures = []

        self._min_cluster_size = min_cluster_size
        self._credits = credits

    def add_im(self, im: model.Image) -> id.Image:
        weighted_medoids = self._signature(im)
        discrete_signature = self._discrete_signature(weighted_medoids)

        self._discrete_signatures[im.id] = discrete_signature
        self._ims[im.id] = im

    def im(self, id: id.Image) -> model.Image:
        return self._ims[id]

    def similar_ims(self, im: model.Image, n: int) -> List[id.Image]:
        pass

    # The sum of weights of all weighted medoids should approximate 1.
    def _signature(self, im: model.Image) -> List[WeightedMedoid]:
        features = self._extractor.features(im.mat)
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self._min_cluster_size)
        labels = clusterer.fit_predict(features)

        centroids = dict()
        cluster_sizes = dict()
        used_features_cnt = 0

        # O(features_count)
        for i in range(len(labels)):
            if labels[i] < 0:
                continue

            used_features_cnt += 1

            if labels[i] not in centroids:
                centroids[labels[i]] = features[i]
                cluster_sizes[labels[i]] = 1
                continue

            centroids[labels[i]] += features[i]
            cluster_sizes[labels[i]] += 1

        # O(labels_count)
        for label in centroids.keys():
            centroids[label] /= cluster_sizes[label]

        # Find the medoids.
        medoids = dict()
        medoid_dists = dict()
        # O(features_count)
        for i in range(len(labels)):
            if labels[i] < 0:
                continue

            dist = linalg.norm(centroids[labels[i]] - features[i])
            if labels[i] not in medoid_dists or medoid_dists[labels[i]] > dist:
                medoids[labels[i]] = features[i]
                medoid_dists[labels[i]] = dist

        # Find the weighted medoids.
        weighted_medoids = []

        for label in medoids.keys():
            weight = cluster_sizes[label] / used_features_cnt
            weighted_medoid = WeightedMedoid(medoid=medoids[label], weight=weight)

            weighted_medoids.append(weighted_medoid)

        return weighted_medoids

    def _discrete_signature(self, signature: List[WeightedMedoid]) -> List[DiscreteHistogram]:
        # The more credits, the better the accuracy.
        left = self._credits
        hists = [None] * len(signature)

        for i in range(len(signature)):
            medoid = signature[i]
            # Handle credits.
            credits = math.floor(self._credits * medoid.weight)

            if i == len(signature)-1:
                credits = left

            left -= credits

            hists[i] = self._discrete_histogram(medoid, credits)

        return hists

    def _discrete_histogram(self, medoid: WeightedMedoid, credits: int) -> DiscreteHistogram:
        total = 0.0

        for val in medoid.medoid:
            total += val

        left = credits
        hist = [0] * len(medoid.medoid)

        for i in range(len(medoid.medoid)):
            ratio = medoid.medoid[i]/total
            hist[i] = math.floor(credits * ratio)
            if i == len(medoid.medoid) - 1:
                hist[i] = left

            left -= hist[i]

        return hist
