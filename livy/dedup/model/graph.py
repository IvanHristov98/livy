from typing import NamedTuple, List, Dict


class Node(NamedTuple):
    resource: int


class Edge(NamedTuple):
    to: int
    # The graph should be directed and weakly connected.
    # This is a helper field that says whether an edge is present only in the directed graph.
    strong: bool
    cost: int


class Graph:
    # Invariant states that the number of adj_lists is equal to the number of nodes.
    adj_list: Dict[int, List[Edge]]
    nodes: Dict[int, Node]

    def __init__(self) -> None:
        self.adj_list = dict()
        self.nodes = dict()

    def add_node(self, idx: int, resource: int) -> None:
        self.adj_list[idx] = []
        self.nodes[idx] = Node(resource=resource)

    def add_edge(self, origin: int, dest: int, cost: float) -> None:
        self.adj_list[origin].append(Edge(to=dest, strong=True, cost=cost))
        # Required for weak connectivity.
        self.adj_list[dest].append(Edge(to=origin, strong=False, cost=cost))


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
