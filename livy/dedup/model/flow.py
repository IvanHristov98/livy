from typing import NamedTuple, Dict, List

from livy.dedup.model.graph import Graph
from livy.dedup.model.spanningtree import SpanningTreeNode


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


# The provided arc is oriented.
def find_unoriented_flow_var(flow_vars: Dict[int, List[FlowEdge]], origin: int, dest: int) -> int:
    if origin in flow_vars:
        for flow_edge in flow_vars[origin]:
            if flow_edge.dest == dest:
                return flow_edge.val

    if dest in flow_vars:
        for flow_edge in flow_vars[dest]:
            if flow_edge.dest == origin:
                return flow_edge.val

    # Should be considered a programming error.
    raise Exception("non existing flow variable")
