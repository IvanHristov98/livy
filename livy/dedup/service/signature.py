from typing import List, NamedTuple

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


class SignatureService(service.Service):
    _extractor: service.Extractor

    def __init__(self, extractor: service.Extractor) -> None:
        self._extractor = extractor

    def add_im(self, im: model.Image) -> id.Image:
        weighted_medoids = self._signature(im)

        print(len(weighted_medoids))

    def im(self, id: id.Image) -> model.Image:
        return self._store[id]

    def similar_ims(self, im: model.Image, n: int) -> List[id.Image]:
        pass

    def _signature(self, im: model.Image) -> List[WeightedMedoid]:
        features = self._extractor.features(im.mat)
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
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
            weight = cluster_sizes[labels[i]] / used_features_cnt
            weighted_medoid = WeightedMedoid(medoid=medoids[label], weight=weight)

            weighted_medoids.append(weighted_medoid)

        return weighted_medoids
