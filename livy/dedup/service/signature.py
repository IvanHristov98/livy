from typing import List, NamedTuple, Dict, Set, Tuple
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
    # weight is an unsigned integer.
    weight: int


class Node(NamedTuple):
    resource: int


class Edge(NamedTuple):
    to: int
    # The graph should be directed and weakly connected.
    # This is a helper field that says whether an edge is present only in the directed graph.
    strong: bool
    cost: int


class Graph(NamedTuple):
    # Invariant states that the number of adj_lists is equal to the number of nodes.
    adj_list: List[List[Edge]]
    nodes: List[Node]


class SpanningTreeNode:
    idx: int
    # The int corresponds to the edge index in the graph.
    children: List[Tuple[int, 'SpanningTreeNode']]

    def __init__(self, idx: int, children: List['SpanningTreeNode']) -> None:
        self.idx = idx
        self.children = children


class FlowEdge(NamedTuple):
    origin: int
    dest: int
    val: int


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
        weighted_medoids = self._signature(im)

        self._signatures[im.id] = weighted_medoids
        self._ims[im.id] = im

    def im(self, id: id.Image) -> model.Image:
        return self._ims[id]

    def similar_ims(self, im: model.Image, n: int) -> List[id.Image]:
        signature = self._signature(im)

        for im_id in self._signatures.keys():
            other_signature = self._signatures[im_id]

            graph = self._build_graph(signature, other_signature)



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
    def _build_graph(self, supplier: List[WeightedMedoid], consumer: List[WeightedMedoid]) -> Graph:
        graph = Graph(adj_list=[], nodes=[])
        idx = 1

        # Number of clusters is usually small (e.g. 10).
        # Hence number of iterations is around 5 * 400 = 2000 nodes.
        for medoid in supplier:
            graph.adj_list.append([])
            graph.nodes.append(Node(resource=medoid.weight))

            idx += 1

        for i in range(len(consumer)):
            consumer_medoid = consumer[i]

            graph.adj_list.append([])
            graph.nodes.append(Node(resource=consumer_medoid.weight))

            for j in range(len(supplier)):
                supplier_medoid = supplier[j]

                dist = linalg.norm(supplier_medoid.medoid - consumer_medoid.medoid)
                graph.adj_list[j].append(Edge(to=idx, strong=True, cost=dist))
                graph.adj_list[idx].append(Edge(to=j, strong=False, cost=dist))

            idx += 1

        return graph

    def _find_optimal_flow_matrix(self, graph: Graph) -> List[List[float]]:
        root = self._spanning_tree(graph)

    def _spanning_tree(self, graph: Graph) -> SpanningTreeNode:
        def _aux_spanning_tree(node: SpanningTreeNode, visited: Set[int]) -> None:
            neighbours = graph.adj_list[node.idx]

            for i in range(len(neighbours)):
                neighbour_edge = neighbours[i]
                if neighbour_edge.to in visited:
                    continue

                child_node = SpanningTreeNode(idx=neighbour_edge.to, children=[])
                node.children.append((i, child_node))

                visited.add(child_node.idx)
                _aux_spanning_tree(child_node, visited)

        root = SpanningTreeNode(idx=self.SRC_IDX, children=[])
        visited = set([self.SRC_IDX])

        _aux_spanning_tree(root, visited)
        return root

    def _assign_flow_values(self, graph: Graph, root: SpanningTreeNode) -> Set[FlowEdge]:
        flows = set([])

        def _aux_assign_flow_values(node: SpanningTreeNode) -> int:
            if len(node.children) == 0:
                # Return required resource.
                return graph.nodes[node.idx]

            supplied = 0.0

            for (edge_idx, node) in node.children:
                # E.g. a child node v has resource -5.
                resource = _aux_assign_flow_values(node)
                edge = graph.adj_list[node.idx][edge_idx]

                # And the edge is going into v from the current node u.
                if edge.strong:
                    # then u sends some of its resource.
                    supplied -= resource
                # and the edge is going from v into u.
                else:
                    # then imaginarily u sends some of its resource in an opposite direction.
                    supplied += resource
                
            return graph.nodes[node.idx] - supplied
