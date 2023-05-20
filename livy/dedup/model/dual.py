from typing import Dict, List

from livy.dedup.model.graph import Graph
from livy.dedup.model.spanningtree import SpanningTreeNode


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


class SlackEdge:
    dest: int
    val: float

    def __init__(self, dest: int, val: float) -> None:
        self.dest = dest
        self.val = val


def find_slack_variables(graph: Graph, dual_vars: Dict[int, float]) -> Dict[int, List[SlackEdge]]:
    EPS = 0.0000000001
    slack_vars = dict()

    for node_idx in graph.adj_list.keys():
        if node_idx not in slack_vars:
            slack_vars[node_idx] = []

        edges = graph.adj_list[node_idx]

        for edge in edges:
            if not edge.strong:
                continue

            slack_val = edge.cost + dual_vars[node_idx] - dual_vars[edge.to]

            # We're working with floats.
            if abs(slack_val) > EPS:
                slack_vars[node_idx].append(SlackEdge(dest=edge.to, val=slack_val))

    return slack_vars
