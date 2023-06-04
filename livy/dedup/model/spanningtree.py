from typing import List, Set

from livy.dedup.model.graph import Graph

_SRC_IDX = 0


class SpanningTreeEdge:
    to: 'SpanningTreeNode'
    # The index corresponds to the edge index in the graph.
    # Allows quick lookup.
    edge_idx: int

    def __init__(self, to: 'SpanningTreeNode', edge_idx: int) -> None:
        self.to = to
        self.edge_idx = edge_idx


class SpanningTreeNode:
    idx: int
    children: List[SpanningTreeEdge]

    def __init__(self, idx: int, children: List[SpanningTreeEdge]) -> None:
        self.idx = idx
        self.children = children


def spanning_tree(graph: Graph) -> SpanningTreeNode:
    def _aux_spanning_tree(node: SpanningTreeNode, visited: Set[int]) -> None:
        neighbours = graph.adj_list[node.idx]

        for i in range(len(neighbours)):
            neighbour_edge = neighbours[i]
            if neighbour_edge.to in visited:
                continue

            child_node = SpanningTreeNode(idx=neighbour_edge.to, children=[])
            node.children.append(SpanningTreeEdge(to=child_node, edge_idx=0))

            visited.add(child_node.idx)
            _aux_spanning_tree(child_node, visited)

    root = SpanningTreeNode(idx=_SRC_IDX, children=[])
    visited = set([_SRC_IDX])

    _aux_spanning_tree(root, visited)
    return root
