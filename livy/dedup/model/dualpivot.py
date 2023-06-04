from typing import List, Dict, Set, Tuple

from livy.dedup.model.graph import Graph, Edge
from livy.dedup.model.spanningtree import SpanningTreeNode, SpanningTreeEdge
from livy.dedup.model.flow import assign_flow_values, FlowEdge
from livy.dedup.model.dual import assign_dual_variables, find_slack_variables, SlackEdge
from livy.dedup.model.state import SimplexState


class DualUnboundedError(Exception):
    """
    DualUnboundedError is thrown whenever pivoting the graph is unbounded.
    """


def dual_pivot(graph: Graph, root: SpanningTreeNode) -> SimplexState:
    flow_vars = assign_flow_values(graph, root)
    dual_vars = assign_dual_variables(graph, root)
    slack_vars = find_slack_variables(graph, dual_vars)

    while True:
        # print("===================")
        (breaking_origin, breaking_edge) = (-1, None)

        for origin in flow_vars.keys():
            for edge in flow_vars[origin]:
                if breaking_edge is None or edge.val <= breaking_edge.val:
                    (breaking_origin, breaking_edge) = (origin, edge)
        
        if breaking_edge.val >= 0:
            break

        # print("breaker", breaking_origin, breaking_edge.dest, "with val", breaking_edge.val)

        c1, c0 = _split_tree(graph, root, (breaking_origin, breaking_edge.dest))

        # Find direction of breaking edge.
        points_to_c0 = _points_to_c0(graph, breaking_origin, breaking_edge, c1, c0)

        # Find the opposing edge of the breaking one that is with the smallest slack variable.
        edges_between_sets = _edges_between_disjoint_sets(graph, c1, c0)

        (min_edge, min_slack_val) = _find_replacing_edge(
            breaking_origin, breaking_edge, points_to_c0, slack_vars, edges_between_sets,
        )
        (new_origin, new_edge) = min_edge

        # Modify basis (aka spanning tree).
        root = _replace_edge(graph, root, (breaking_origin, breaking_edge.dest), (new_origin, new_edge.to))

        # Change flow variables.
        flow_vars = assign_flow_values(graph, root)

        # Change dual variables.
        # Assignment could be more optimal, but since the number of nodes is always <100 it shouldn't be
        # a performance issue.
        dual_vars = assign_dual_variables(graph, root)

        # Change slack variables.
        _update_slack_vars(min_slack_val, points_to_c0, slack_vars, edges_between_sets)

    return SimplexState(root, flow_vars, dual_vars, slack_vars)


# _split_tree splits the nodes of a given spanning tree.
# The returned sets of node indices are disjoint.
# The first set contains the root.
def _split_tree(graph: Graph, root: SpanningTreeNode, breaking_edge: Tuple[int, int]) -> Tuple[Set[int], Set[int]]:
    c1 = set()

    def _find_root_set(curr: SpanningTreeNode) -> None:
        c1.add(curr.idx)
        (break_origin, break_dest) = breaking_edge

        for child in curr.children:
            if curr.idx == break_origin and child.to.idx == break_dest:
                continue
            
            if curr.idx == break_dest and child.to.idx == break_origin:
                continue

            _find_root_set(child.to)

    _find_root_set(root)

    c0 = set()

    for node in graph.nodes.keys():
        if node not in c1:
            c0.add(node)

    return (c1, c0)


def _points_to_c0(graph: Graph, breaking_origin: int, breaking_edge: FlowEdge, c1: Set[int], c0: Set[int]) -> bool:
    for other_edge in graph.adj_list[breaking_origin]:
        if other_edge.to != breaking_edge.dest:
            continue

        # It is guaranteed that this statement would be reached because we know that the edge exists.
        if (breaking_origin in c1 and other_edge.strong) or (breaking_origin in c0 and not other_edge.strong):
            return True
        else:
            return False
    
    raise Exception("algorithm was supposed to find edge when finding direction of removed edge")


# The returned edges are always guaranteed to be from c1 to c0.
# This includes weak edges.
def _edges_between_disjoint_sets(graph: Graph, c1: Set[int], c0: Set[int]) -> List[Tuple[int, Edge]]:
    edges = []

    for c1_node in c1:
        # We take advantage of the fact that the graph is weakly connected
        # (i.e. there is a weak edge for every strong edge).
        for edge in graph.adj_list[c1_node]:
            if edge.to not in c0:
                continue

            edges.append((c1_node, edge))

    return edges


def _find_replacing_edge(
        breaking_origin: int,
        breaking_edge: FlowEdge,
        breaking_edge_points_to_c0: bool,
        slack_vars: Dict[int, List[SlackEdge]],
        edges_between_sets: List[Tuple[int, Edge]],
    ) -> Tuple[Tuple[int, Edge], float]:
    min_slack_val = 0.0
    min_edge = None

    for (origin, edge) in edges_between_sets:
        # Check for same direction.
        # The edges are always guaranteed to be from c1 to c0.
        if (edge.strong and breaking_edge_points_to_c0) or (not edge.strong and not breaking_edge_points_to_c0):
            # We don't care about edges between c1 and c0 that are with the same direction as
            # the breaking edge.
            continue

        # If not found in slack_vars we assume that the value is 0,
        # because the graph basis should be dual feasible.
        slack_var = 0.0

        for slack_edge in slack_vars[origin]:
            if slack_edge.dest != edge.to:
                continue
            slack_var = slack_edge.val

        if min_edge is None or min_slack_val > slack_var:
            min_slack_val = slack_var
            min_edge = (origin, edge)
    
    if min_edge is None:
        raise DualUnboundedError(
            f"graph is dual unbounded: no opposite edges found for {breaking_origin}-{breaking_edge.dest}",
        )

    return min_edge, min_slack_val


def _replace_edge(
        graph: Graph,
        root: SpanningTreeNode, 
        replaced_edge: Tuple[int, int], 
        new_edge: Tuple[int, int],
    ) -> SpanningTreeNode:
    adj_list: Dict[int, Set[int]] = dict()

    def _build_tree_adj_list(curr: SpanningTreeNode):
        for child in curr.children:
            if curr.idx not in adj_list:
                adj_list[curr.idx] = set()
            adj_list[curr.idx].add(child.to.idx)

            if child.to.idx not in adj_list:
                adj_list[child.to.idx] = set()
            adj_list[child.to.idx].add(curr.idx)

            _build_tree_adj_list(child.to)

    _build_tree_adj_list(root)

    adj_list[replaced_edge[0]].remove(replaced_edge[1])
    adj_list[replaced_edge[1]].remove(replaced_edge[0])

    adj_list[new_edge[0]].add(new_edge[1])
    adj_list[new_edge[1]].add(new_edge[0])

    nodes: Dict[int, SpanningTreeNode] = dict()

    for node_idx in graph.nodes.keys():
        nodes[node_idx] = SpanningTreeNode(node_idx, children=[])

    added = set()

    def _build_new_tree(curr: SpanningTreeNode):
        idx_map: Dict[int, int] = dict()
        for i in range(len(graph.adj_list[curr.idx])):
            idx_map[graph.adj_list[curr.idx][i].to] = i

        for child_idx in adj_list[curr.idx]:
            if curr.idx in added and child_idx in added:
                continue

            nodes[curr.idx].children.append(SpanningTreeEdge(to=nodes[child_idx], edge_idx=idx_map[child_idx]))
            added.add(curr.idx)
            added.add(child_idx)

            _build_new_tree(nodes[child_idx])

    _build_new_tree(nodes[root.idx])
    return nodes[root.idx]


def _update_slack_vars(
        min_slack_val: float,
        breaking_edge_points_to_c0: bool,
        slack_vars: Dict[int, List[SlackEdge]],
        edges_between_sets: List[Tuple[int, Edge]],
) -> None:
    # Change slack variables.
    for (origin, edge) in edges_between_sets:
        # Slack variables are defined only for strong edges.
        strong_origin = origin
        strong_dest = edge.to

        if not edge.strong:
            strong_origin = edge.to
            strong_dest = origin
        
        # print("attempt", strong_origin, strong_dest)
        found = False

        for i in range(len(slack_vars[strong_origin])):
            if slack_vars[strong_origin][i].dest != strong_dest:
                continue

            found = True

            # Check for same direction.
            # The edges are always guaranteed to be from c1 to c0.
            if (edge.strong and breaking_edge_points_to_c0) or (not edge.strong and not breaking_edge_points_to_c0):
                slack_vars[strong_origin][i].val -= min_slack_val
            else:
                slack_vars[strong_origin][i].val += min_slack_val
            break
        
        if not found:
            # If the value didn't exist beforehand, then it means that the value was 0.
            # min_slack_val is always >= 0. Hence to guarantee dual feasibility it must be added.
            # print("adding", strong_origin, strong_dest)
            slack_vars[strong_origin].append(SlackEdge(dest=strong_dest, val=min_slack_val))
        
        # There is no need for deletion of slack vars that have become 0.
