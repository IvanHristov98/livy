from typing import Dict, Set, List, Tuple

from livy.dedup.model.graph import Graph, Edge
from livy.dedup.model.spanningtree import SpanningTreeNode, SpanningTreeEdge
from livy.dedup.model.state import SimplexState
from livy.dedup.model.flow import assign_flow_values, FlowEdge
from livy.dedup.model.dual import (
    assign_dual_variables, 
    find_slack_variables,
    SlackEdge,
)


class PrimalUnboundedError(Exception):
    """
    PrimalUnboundedError is thrown whenever pivoting the graph is unbounded.
    """


def primal_pivot(graph: Graph, root: SpanningTreeNode) -> SimplexState:
    tree_adj_list = _spanning_tree_to_weak_adj_list(root)
    flow_vars = assign_flow_values(graph, root)
    dual_vars = assign_dual_variables(graph, root)
    slack_vars = find_slack_variables(graph, dual_vars)

    while True:
        # It is guaranteed to be a strong edge.
        (new_origin, new_slack_edge) = _find_min_slack_var(slack_vars)

        if new_slack_edge.val >= 0:
            break

        _replace_edge(graph, flow_vars, tree_adj_list, new_origin, new_slack_edge)
        root = _weak_adj_list_to_spanning_tree(graph, tree_adj_list, root.idx)

        # TODO: Optimize the update these parameters.
        tree_adj_list = _spanning_tree_to_weak_adj_list(root)
        flow_vars = assign_flow_values(graph, root)
        dual_vars = assign_dual_variables(graph, root)
        slack_vars = find_slack_variables(graph, dual_vars)

    return SimplexState(root, flow_vars, dual_vars, slack_vars)


def _spanning_tree_to_weak_adj_list(root: SpanningTreeNode) -> Dict[int, Set[int]]:
    adj_list : Dict[int, Set[int]] = dict()

    def _aux_spanning_tree_to_weak_adj_list(curr: SpanningTreeNode):
        if curr.idx not in adj_list:
            adj_list[curr.idx] = set()
        
        for edge in curr.children:
            adj_list[curr.idx].add(edge.to.idx)

            if edge.to.idx not in adj_list:
                adj_list[edge.to.idx] = set()
            
            adj_list[edge.to.idx].add(curr.idx)

            _aux_spanning_tree_to_weak_adj_list(edge.to)

    _aux_spanning_tree_to_weak_adj_list(root)

    return adj_list


def _weak_adj_list_to_spanning_tree(
    graph: Graph, 
    tree_adj_list: Dict[int, Set[int]], 
    root_idx: int,
) -> SpanningTreeNode:
    visited : Set[int] = set()

    def _aux_weak_adj_list_from_spanning_tree(curr_idx: int) -> SpanningTreeNode:
        curr = SpanningTreeNode(idx=curr_idx, children=[])
        visited.add(curr_idx)

        for child_idx in tree_adj_list[curr_idx]:
            if child_idx in visited:
                continue

            child = _aux_weak_adj_list_from_spanning_tree(child_idx)
            edge_idx = _find_edge_idx(graph, curr_idx, child_idx)
            curr.children.append(SpanningTreeEdge(to=child, edge_idx=edge_idx))

        return curr

    return _aux_weak_adj_list_from_spanning_tree(root_idx)


def _find_edge_idx(graph: Graph, origin: int, dest: int) -> int:
    for i in range(len(graph.adj_list[origin])):
        if graph.adj_list[origin][i].to == dest:
            return i

    raise Exception(f"edge not found {origin}-{dest}")


def _find_min_slack_var(slack_vars: Dict[int, List[SlackEdge]]) -> Tuple[int, SlackEdge]:
    min_origin = -1
    min_slack_edge = None

    for origin in slack_vars.keys():
        for slack_edge in slack_vars[origin]:
            if min_slack_edge is None or min_slack_edge.val > slack_edge.val:
                min_origin = origin
                min_slack_edge = slack_edge

    return (min_origin, min_slack_edge)


def _replace_edge(
    graph: Graph,
    flow_vars: Dict[int, List[FlowEdge]],
    tree_adj_list: Dict[int, Set[int]],
    new_origin: int,
    new_slack_edge: SlackEdge,
) -> Dict[int, Set[int]]:
    # Add the new edge to introduce a cycle.
    _add_edge_to_tree_adj_list(tree_adj_list, new_origin, new_slack_edge.dest)

    # Find the added cycle.
    cycle = _find_added_cycle(tree_adj_list, new_origin)

    # Find the edge with the smallest flow var that is opposite of the added edge.
    new_edge = (new_origin, new_slack_edge.dest)
    (min_origin, min_dest) = _find_opposite_edge_with_min_flow(graph, flow_vars, cycle, new_edge)

    _remove_edge_from_tree_adj_list(tree_adj_list, min_origin, min_dest)


def _add_edge_to_tree_adj_list(tree_adj_list: Dict[int, Set[int]], new_origin: int, new_dest: int) -> None:
    if new_origin not in tree_adj_list:
        tree_adj_list[new_origin] = set()

    tree_adj_list[new_origin].add(new_dest)

    if new_dest not in tree_adj_list:
        tree_adj_list[new_dest] = set()

    tree_adj_list[new_dest].add(new_origin)


# _find_added_cycle returns a path of unique elements that form a cycle starting from start.
def _find_added_cycle(tree_adj_list: Dict[int, Set[int]], start: int) -> List[int]:
    path: List[int] = []
    visited : Set[int] = set()
    cycle: List[int] = []

    def _detect_cycle(curr: int) -> None:
        path.append(curr)
        visited.add(curr)

        for dest in tree_adj_list[curr]:
            # We know that there is exactly one cycle in the modified tree graph.
            # It should contain at least 3 unique nodes.
            if dest == start and len(path) > 2:
                for i in range(len(path)):
                    cycle.append(path[i])
                break

            if dest not in visited:
                _detect_cycle(dest)

        path.pop()

    _detect_cycle(start)
    return cycle


def _find_opposite_edge_with_min_flow(
    graph: Graph,
    flow_vars: Dict[int, List[FlowEdge]],
    cycle: List[int],
    new_edge: Tuple[int, int],
) -> Tuple[int, int]:
    # Cycles could be traversed in 2 directions and the new_edge is always guaranteed to be a strong one.
    # Cycles always start from the origin of the new edge.
    # Hence if the second cycle element is the dest of the new edge, then the edges opposite to it should
    # not be strong.
    should_be_strong = False

    # If the last cycle element is the dest of the new edge, then we're traversing the cycle in a reverse direction.
    # Hence the searched for edges should be strong.
    if cycle[len(cycle)-1] == new_edge[1]:
        should_be_strong = True

    min_flow_val = -1
    min_edge = None
    total_cost = 0

    edge = _find_edge(graph, new_edge[0], new_edge[1])

    # Always skip the first edge as it is the added.
    for i in range(len(cycle)):
        origin = cycle[i]
        dest = cycle[(i+1)%len(cycle)]

        if (origin == new_edge[0] and dest == new_edge[1]) or (origin == new_edge[1] and dest == new_edge[0]):
            continue

        flow_val = _find_unoriented_flow_var(flow_vars, origin, dest)
        edge = _find_edge(graph, origin, dest)

        if edge.strong == should_be_strong and (min_edge is None or min_flow_val > flow_val):
            min_flow_val = flow_val
            min_edge = (origin, dest)
        
        total_cost += edge.cost

    if min_edge is None and total_cost < 0:
        raise PrimalUnboundedError(f"encountered unbounded cycle {cycle}")
    
    if min_edge is None:
        raise Exception("no opposite flow edges were found during primal pivot")

    return min_edge


# The provided arc is oriented.
def _find_unoriented_flow_var(flow_vars: Dict[int, List[FlowEdge]], origin: int, dest: int) -> int:
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


def _find_edge(graph: Graph, origin: int, dest: int) -> Edge:
    for edge in graph.adj_list[origin]:
        if edge.to == dest:
            return edge
    
    raise Exception("working with non-existing edge")


def _remove_edge_from_tree_adj_list(tree_adj_list: Dict[int, Set[int]], origin: int, dest: int) -> None:
    if origin in tree_adj_list:
        tree_adj_list[origin].remove(dest)

    if dest in tree_adj_list:
        tree_adj_list[dest].remove(origin)
