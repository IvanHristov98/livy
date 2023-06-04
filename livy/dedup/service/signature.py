from typing import List, NamedTuple, Dict, Set
import math

import hdbscan
import numpy as np
import numpy.linalg as linalg

import livy.id as id
import livy.model as model
import livy.dedup.model as dedupmodel
import livy.dedup.service as service


class WeightedMedoid(NamedTuple):
    # medoid is the most centrally located element of a cluster.
    medoid: np.ndarray
    # weight is an unsigned integer.
    weight: int


class SignatureService(service.Service):
    SRC_IDX = 0

    # Hyper-parameters.
    _min_cluster_size: int
    _credits: int

    _extractor: service.Extractor
    _signatures: Dict[id.Image, WeightedMedoid]
    _ims: Dict[id.Image, model.Image]

    def __init__(self, extractor: service.Extractor, min_cluster_size: int = 5, credits: int = 1000000) -> None:
        self._extractor = extractor
        self._signatures = dict()
        self._ims = dict()

        self._min_cluster_size = min_cluster_size
        self._credits = credits

    def add_im(self, im: model.Image) -> id.Image:
        print(f"loaded image {im.id}")

        weighted_medoids = self._signature(im)

        self._signatures[im.id] = weighted_medoids
        self._ims[im.id] = im

    def im(self, id: id.Image) -> model.Image:
        return self._ims[id]

    def similar_ims(self, im: model.Image, n: int) -> List[id.Image]:
        signature = self._signature(im)

        class Score(NamedTuple):
            id: id.Image
            score: float

        top_ims: List[Score] = []

        for i in range(n+1):
            top_ims.append(Score(id=id.NoImage, score=0.0))

        for other_im_id in self._signatures.keys():
            other_signature = self._signatures[other_im_id]

            graph = self._build_graph(signature, other_signature)
            try:
                simplex_state = dedupmodel.network_simplex(graph)
                cost = dedupmodel.total_cost(graph, simplex_state)
            except Exception as e:
                print(f"Exception encountered for comparison between {other_im_id} and {im.id}: {str(e)}; Skipping;")
                continue

            # The bigger the cost the further the image is.
            # Hence a invert is necessary.
            score = 1/cost

            top_ims[n] = Score(id=other_im_id, score=score)

            # An insert sort is used to order small number of items fast.
            # Other sorts would be slow.
            for i in range(n-1, -1, -1):
                if top_ims[i].score > score:
                    break

                top_ims[i+1], top_ims[i] = top_ims[i], top_ims[i+1]

        return top_ims[:n]

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

        # Calculating weight.
        # The more credits, the better the accuracy.
        left = self._credits
        weighted_medoids = []

        for label in medoids.keys():
            weight = cluster_sizes[label] / used_features_cnt
            discrete_weight = math.floor(self._credits * weight)

            if len(weighted_medoids) == len(medoids.keys()) - 1:
                discrete_weight = left
            
            left -= discrete_weight

            weighted_medoid = WeightedMedoid(medoid=medoids[label], weight=discrete_weight)
            weighted_medoids.append(weighted_medoid)

        return weighted_medoids

    # The number of credits in the supplier signature should be equal ot those in the consumer signature.
    def _build_graph(
            self, 
            supplying_medoids: List[WeightedMedoid], 
            demanding_medoids: List[WeightedMedoid],
        ) -> dedupmodel.Graph:
        graph = dedupmodel.Graph()
        idx = 0

        # Number of clusters is usually small (e.g. 10).
        for medoid in supplying_medoids:
            graph.add_node(idx, resource=medoid.weight)
            idx += 1

        for medoid in demanding_medoids:
            graph.add_node(idx, resource=-medoid.weight)

            for k in range(len(supplying_medoids)):
                supplying_medoid = supplying_medoids[k]
                dist = linalg.norm(supplying_medoid.medoid - medoid.medoid)
                graph.add_edge(origin=k, dest=idx, cost=dist)

            idx += 1

        return graph
