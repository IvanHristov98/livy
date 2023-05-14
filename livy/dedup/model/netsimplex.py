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


# FlowEdge contains the flow that is sent along an edge in the spanning tree.
# Values are set with regard to the direction of graph edges.
class FlowEdge(NamedTuple):
    dest: int
    val: int


def assign_flow_values(graph: Graph, root: SpanningTreeNode) -> Dict[int, List[FlowEdge]]:
    flows = dict()

    def _aux_assign_flow_values(node: SpanningTreeNode) -> int:
        if len(node.children) == 0:
            # Return required resource.
            return graph.nodes[node.idx].resource

        supply = graph.nodes[node.idx].resource
        flows[node.idx] = []

        for edge in node.children:
            # E.g. a child node v has resource -5.
            resource = _aux_assign_flow_values(edge.to)
            graph_edge = graph.adj_list[node.idx][edge.edge_idx]

            # And the edge is going into v from the current node u.
            if graph_edge.strong:
                # then u sends some of its resource.
                flows[node.idx].append(FlowEdge(dest=edge.to.idx, val=-resource))
            # and the edge is going from v into u.
            else:
                flows[node.idx].append(FlowEdge(dest=edge.to.idx, val=resource))

            supply += resource

        return supply

    _aux_assign_flow_values(root)
    return flows


# TODO: Raise exception on negative balance.


def assign_dual_variables(graph: Graph, root: SpanningTreeNode) -> Dict[int, float]:
    dual_vars = dict()

    def _aux_assign_dual_variables(node: SpanningTreeNode, offset: float) -> None:
        dual_vars[node.idx] = offset

        for edge in node.children:
            graph_edge = graph.adj_list[node.idx][edge.edge_idx]

            if graph_edge.strong:
                _aux_assign_dual_variables(edge.to, offset + graph_edge.cost)
            else:
                _aux_assign_dual_variables(edge.to, offset - graph_edge.cost)            

    _aux_assign_dual_variables(root, 0)
    return dual_vars
