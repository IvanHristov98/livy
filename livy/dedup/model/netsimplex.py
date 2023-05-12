from typing import NamedTuple, List, Dict, Tuple, Set


_SRC_IDX = 0


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


class FlowEdge(NamedTuple):
    dest: int
    val: int


def assign_flow_values(graph: Graph, root: SpanningTreeNode) -> Dict[int, List[FlowEdge]]:
    flows = dict()

    def _aux_assign_flow_values(node: SpanningTreeNode) -> int:
        if len(node.children) == 0:
            # Return required resource.
            return graph.nodes[node.idx].resource

        supplied = 0.0
        flows[node.idx] = []

        for edge in node.children:
            # E.g. a child node v has resource -5.
            resource = _aux_assign_flow_values(edge.to)
            graph_edge = graph.adj_list[node.idx][edge.edge_idx]

            # And the edge is going into v from the current node u.
            if graph_edge.strong:
                # then u sends some of its resource.
                flows[node.idx].append(FlowEdge(dest=edge.to.idx, val=-resource))
                supplied -= resource
            # and the edge is going from v into u.
            else:
                # TODO:
                pass
                # then imaginarily u sends some of its resource in an opposite direction.
                # flows[node.idx].append(FlowEdge(dest=edge.to.idx, val=resource))
                # supplied += resource

        return graph.nodes[node.idx].resource - supplied

    _aux_assign_flow_values(root)
    return flows
