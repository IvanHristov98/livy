from livy.dedup.model.graph import Graph, copy_graph_with_zero_costs
from livy.dedup.model.state import SimplexState
from livy.dedup.model.spanningtree import spanning_tree
from livy.dedup.model.dualpivot import dual_pivot
from livy.dedup.model.primalpivot import primal_pivot


# network_simplex performs a dual-first network simplex method to find an optimal solution
# to a minimum-cost network flow problem.
#
# The provided graph must be balanced (in terms of supply and demand).
# If the optimization is unbounded, respective exceptions are thrown.
def network_simplex(graph: Graph) -> SimplexState:
    free_graph = copy_graph_with_zero_costs(graph)
    root = spanning_tree(free_graph)

    state = dual_pivot(free_graph, root)
    return primal_pivot(graph, state.root)
