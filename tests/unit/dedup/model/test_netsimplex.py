import unittest
from typing import List, Dict

import livy.dedup.model as model


class TestSpanningTree(unittest.TestCase):
    _graph: model.Graph

    def setUp(self) -> None:
        self._graph = model.Graph()
    
    def test_single_node(self) -> None:
        self._graph.add_node(0, 0)

        root = model.spanning_tree(self._graph)

        self.assertEqual(root.idx, 0)
        self.assertEqual(root.children, [])

    def test_unidirectional_bipartite_graph(self) -> None:
        self._graph = bipartite_unidirectional_graph()

        root = model.spanning_tree(self._graph)

        node2 = model.SpanningTreeNode(idx=2, children=[])
        node4 = model.SpanningTreeNode(idx=4, children=[model.SpanningTreeEdge(to=node2, edge_idx=0)])
        node1 = model.SpanningTreeNode(idx=1, children=[model.SpanningTreeEdge(to=node4, edge_idx=0)])
        node3 = model.SpanningTreeNode(idx=3, children=[model.SpanningTreeEdge(to=node1, edge_idx=0)])
        expected_root = model.SpanningTreeNode(idx=0, children=[model.SpanningTreeEdge(to=node3, edge_idx=0)])

        assert_spanning_trees(self, root, expected_root)


class TestAssignFlowValues(unittest.TestCase):
    _graph: model.Graph
    _root: model.SpanningTreeNode

    def setUp(self) -> None:
        self._graph = bipartite_unidirectional_graph()
        self._root = model.SpanningTreeNode(idx=0, children=[])

    def test_strong_edge_to_demanding_node(self) -> None:
        node3 = model.SpanningTreeNode(idx=3, children=[])
        self._root.children.append(model.SpanningTreeEdge(to=node3, edge_idx=0))

        flows = model.assign_flow_values(self._graph, self._root)

        self.assertIn(0, flows)
        self.assertIn(model.FlowEdge(dest=3, val=6), flows[0])

    def test_weak_edge_to_demanding_node(self) -> None:
        self._graph.add_node(10, -10)
        self._graph.add_edge(10, 1, cost=10)

        node10 = model.SpanningTreeNode(idx=10, children=[])
        node1 = model.SpanningTreeNode(idx=1, children=[])

        node1.children.append(model.SpanningTreeEdge(to=node10, edge_idx=2))

        flows = model.assign_flow_values(self._graph, node1)

        self.assertIn(1, flows)
        self.assertIn(model.FlowEdge(dest=10, val=-10), flows[1])

    def test_strong_edge_to_supplying_node(self) -> None:
        self._graph.add_node(10, -10)
        self._graph.add_edge(10, 1, cost=10)

        node10 = model.SpanningTreeNode(idx=10, children=[])
        node1 = model.SpanningTreeNode(idx=1, children=[])

        node10.children.append(model.SpanningTreeEdge(to=node1, edge_idx=0))

        flows = model.assign_flow_values(self._graph, node10)

        self.assertIn(10, flows)
        self.assertIn(model.FlowEdge(dest=1, val=-2), flows[10])

    def test_weak_edge_to_supplying_node(self) -> None:
        node3 = model.SpanningTreeNode(idx=3, children=[])
        node3.children.append(model.SpanningTreeEdge(to=self._root, edge_idx=0))

        flows = model.assign_flow_values(self._graph, node3)

        self.assertIn(3, flows)
        self.assertIn(model.FlowEdge(dest=0, val=1), flows[3])

    def test_resource_subtraction(self) -> None:
        graph = infeasible_balanced_graph()
        root = infeasible_graph_full_spanning_tree()

        flows = model.assign_flow_values(graph, root)

        self.assertIn(2, flows)
        self.assertIn(model.FlowEdge(dest=0, val=-1), flows[2])
        self.assertIn(model.FlowEdge(dest=1, val=3), flows[2])

        self.assertIn(3, flows)
        self.assertIn(model.FlowEdge(dest=2, val=-5), flows[3])
        self.assertIn(model.FlowEdge(dest=4, val=7), flows[3])

        self.assertIn(5, flows)
        self.assertIn(model.FlowEdge(dest=3, val=3), flows[5])


class TestAssignDualVariables(unittest.TestCase):
    _graph: model.Graph

    def setUp(self) -> None:
        self._graph = infeasible_balanced_graph()

    def test_single_node_spanning_tree(self) -> None:
        node5 = model.SpanningTreeNode(idx=5, children=[])
        dual_vars = model.assign_dual_variables(self._graph, node5)

        self._assert_dual_var(dual_vars, expected_idx=5, expected_val=0)

    def test_full_spanning_tree(self) -> None:
        root = infeasible_graph_full_spanning_tree()
        dual_vars = model.assign_dual_variables(self._graph, root)

        # Comparison with almost equal is necessary because we're working with floats.
        self._assert_dual_var(dual_vars, expected_idx=5, expected_val=0)
        self._assert_dual_var(dual_vars, expected_idx=3, expected_val=1)
        self._assert_dual_var(dual_vars, expected_idx=4, expected_val=-7)
        self._assert_dual_var(dual_vars, expected_idx=2, expected_val=-1)
        self._assert_dual_var(dual_vars, expected_idx=0, expected_val=0)
        self._assert_dual_var(dual_vars, expected_idx=1, expected_val=4)

    def _assert_dual_var(self, dual_vars: Dict[int, float], expected_idx: int, expected_val: float) -> None:
        EPS = 0.000001

        self.assertIn(expected_idx, dual_vars)
        self.assertAlmostEqual(dual_vars[expected_idx], expected_val, delta=EPS)


class TestFindSlackVariables(unittest.TestCase):
    def test_with_letter_shaped_graph(self) -> None:
        graph = letter_shape_graph()
        # Define dual variables.
        dual_vars = {0: 0, 1: 2, 2: 3, 3: 9, 4: 11, 5: 5}

        slack_vars = model.find_slack_variables(graph, dual_vars)

        self._assert_contains_slack_edge(slack_vars, origin=4, expected_edge=model.SlackEdge(dest=1, val=16))
        self._assert_contains_slack_edge(slack_vars, origin=2, expected_edge=model.SlackEdge(dest=3, val=-5))
        self._assert_contains_slack_edge(slack_vars, origin=2, expected_edge=model.SlackEdge(dest=5, val=7))
        self._assert_contains_slack_edge(slack_vars, origin=5, expected_edge=model.SlackEdge(dest=4, val=-4))

    def _assert_contains_slack_edge(
            self, 
            slack_vars: Dict[int, List[model.SlackEdge]], 
            origin: int, 
            expected_edge: model.SlackEdge,
        ) -> None:
        EPS = 0.000001
        self.assertIn(origin, slack_vars)

        for slack_var in slack_vars[origin]:
            if slack_var.dest == expected_edge.dest:
                self.assertAlmostEqual(slack_var.val, expected_edge.val, delta=EPS)
                return

        self.assertTrue(False, msg=f"slack edge not found for origin {origin} in slack vars {slack_vars}")


class TestDualPivot(unittest.TestCase):
    def test_dual_pivot_on_bounded_graph(self) -> None:
        # See https://youtu.be/ife2d0p4dug?t=3303.
        # Build graph.
        graph = model.Graph()
        # Add nodes from `a` to `h`.
        graph.add_node(0, resource=-4) # a
        graph.add_node(1, resource=5) # b
        graph.add_node(2, resource=1) # c
        graph.add_node(3, resource=1) # d
        graph.add_node(4, resource=2) # e
        graph.add_node(5, resource=0) # f
        graph.add_node(6, resource=3) # g
        graph.add_node(7, resource=2) # h

        # left side
        graph.add_edge(origin=0, dest=1, cost=2)
        graph.add_edge(origin=1, dest=2, cost=3)
        # top x
        graph.add_edge(origin=2, dest=4, cost=1)
        graph.add_edge(origin=4, dest=1, cost=3)
        graph.add_edge(origin=7, dest=4, cost=13)
        graph.add_edge(origin=6, dest=4, cost=10)
        # bot x
        graph.add_edge(origin=3, dest=1, cost=2)
        graph.add_edge(origin=3, dest=0, cost=4)
        graph.add_edge(origin=3, dest=6, cost=7)
        graph.add_edge(origin=5, dest=3, cost=5)
        # bot side
        graph.add_edge(origin=0, dest=5, cost=12)
        # right side
        graph.add_edge(origin=7, dest=6, cost=1)
        graph.add_edge(origin=6, dest=5, cost=2)
        # top side
        graph.add_edge(origin=2, dest=7, cost=10)

        # Build spanning tree that is dual feasible.
        nodes : List[model.SpanningTreeNode] = [None] * 8
        for i in range(0, 8):
            nodes[i] = model.SpanningTreeNode(idx=i, children=[])

        nodes[0].children.append(model.SpanningTreeEdge(to=nodes[1], edge_idx=0))
        nodes[1].children.append(model.SpanningTreeEdge(to=nodes[2], edge_idx=1))
        nodes[1].children.append(model.SpanningTreeEdge(to=nodes[3], edge_idx=3))
        nodes[2].children.append(model.SpanningTreeEdge(to=nodes[4], edge_idx=1))
        nodes[3].children.append(model.SpanningTreeEdge(to=nodes[6], edge_idx=2))
        nodes[6].children.append(model.SpanningTreeEdge(to=nodes[7], edge_idx=2))
        nodes[6].children.append(model.SpanningTreeEdge(to=nodes[5], edge_idx=3))

        state = model.dual_pivot(graph, nodes[0])

        # Assert optimal spanning tree.
        expected_nodes : List[model.SpanningTreeNode] = [None] * 8
        for i in range(0, 8):
            expected_nodes[i] = model.SpanningTreeNode(idx=i, children=[])

        expected_nodes[0].children.append(model.SpanningTreeEdge(to=expected_nodes[3], edge_idx=1))
        expected_nodes[3].children.append(model.SpanningTreeEdge(to=expected_nodes[5], edge_idx=3))
        expected_nodes[5].children.append(model.SpanningTreeEdge(to=expected_nodes[6], edge_idx=2))
        expected_nodes[6].children.append(model.SpanningTreeEdge(to=expected_nodes[7], edge_idx=2))
        expected_nodes[7].children.append(model.SpanningTreeEdge(to=expected_nodes[2], edge_idx=2))
        expected_nodes[2].children.append(model.SpanningTreeEdge(to=expected_nodes[1], edge_idx=0))
        expected_nodes[1].children.append(model.SpanningTreeEdge(to=expected_nodes[4], edge_idx=2))

        assert_spanning_trees(self, state.root, expected_nodes[0])

    def test_dual_pivot_on_unbounded_graph(self) -> None:
        # Build graph.
        graph = model.Graph()
        graph.add_node(0, -1)
        graph.add_node(1, -1)
        graph.add_node(2, 3)

        graph.add_edge(0, 2, 1)
        graph.add_edge(1, 2, 2)

        # Build spanning tree that is dual feasible.
        nodes : List[model.SpanningTreeNode] = [None] * 3
        for i in range(0, 3):
            nodes[i] = model.SpanningTreeNode(idx=i, children=[])
        
        nodes[0].children.append(model.SpanningTreeEdge(to=nodes[2], edge_idx=0))
        nodes[2].children.append(model.SpanningTreeEdge(to=nodes[1], edge_idx=1))

        with self.assertRaises(model.DualUnboundedError):
            model.dual_pivot(graph, nodes[0])


class TestPrimalPivot(unittest.TestCase):
    def test_primal_pivot_on_bounded_graph(self) -> None:
        # See https://youtu.be/zgtY5nGAMgY?t=3147.
        # Build graph.
        graph = model.Graph()

        # Add nodes from `a` to `h`.
        graph.add_node(0, resource=-4) # a
        graph.add_node(1, resource=2) # b
        graph.add_node(2, resource=1) # c
        graph.add_node(3, resource=0) # d
        graph.add_node(4, resource=-4) # e
        graph.add_node(5, resource=5) # f
        graph.add_node(6, resource=-5) # g
        graph.add_node(7, resource=-1) # h
        graph.add_node(8, resource=9) # i
        graph.add_node(9, resource=7) # j
        graph.add_node(10, resource=-10) # k

        # bottom
        graph.add_edge(origin=1, dest=0, cost=1)
        graph.add_edge(origin=2, dest=0, cost=1)
        # mid row
        graph.add_edge(origin=4, dest=5, cost=4)
        graph.add_edge(origin=4, dest=3, cost=3)
        # top
        graph.add_edge(origin=9, dest=8, cost=6)
        graph.add_edge(origin=10, dest=8, cost=3)
        # left side
        graph.add_edge(origin=9, dest=5, cost=1)
        graph.add_edge(origin=5, dest=1, cost=10)
        # right side
        graph.add_edge(origin=3, dest=10, cost=2)
        graph.add_edge(origin=3, dest=2, cost=2)
        # bot and mid rows connector
        graph.add_edge(origin=0, dest=4, cost=5)
        # big triangle sides
        graph.add_edge(origin=5, dest=6, cost=3)
        graph.add_edge(origin=8, dest=7, cost=2)
        graph.add_edge(origin=6, dest=8, cost=5)
        graph.add_edge(origin=7, dest=3, cost=1)
        # reverse small triangle
        graph.add_edge(origin=4, dest=6, cost=1)
        graph.add_edge(origin=6, dest=7, cost=1)
        graph.add_edge(origin=4, dest=7, cost=6)
        # sidelined small triangle side
        graph.add_edge(origin=10, dest=7, cost=4)

        # Build spanning tree that is primal feasible.
        nodes : List[model.SpanningTreeNode] = [None] * 11
        for i in range(0, 11):
            nodes[i] = model.SpanningTreeNode(idx=i, children=[])

        nodes[0].children.append(model.SpanningTreeEdge(to=nodes[4], edge_idx=2))
        nodes[0].children.append(model.SpanningTreeEdge(to=nodes[2], edge_idx=1))
        nodes[0].children.append(model.SpanningTreeEdge(to=nodes[1], edge_idx=0))
        nodes[1].children.append(model.SpanningTreeEdge(to=nodes[5], edge_idx=1))
        nodes[5].children.append(model.SpanningTreeEdge(to=nodes[9], edge_idx=1))
        nodes[4].children.append(model.SpanningTreeEdge(to=nodes[6], edge_idx=3))
        nodes[6].children.append(model.SpanningTreeEdge(to=nodes[8], edge_idx=1))
        nodes[8].children.append(model.SpanningTreeEdge(to=nodes[7], edge_idx=3))
        nodes[7].children.append(model.SpanningTreeEdge(to=nodes[3], edge_idx=1))
        nodes[3].children.append(model.SpanningTreeEdge(to=nodes[10], edge_idx=1))

        state = model.primal_pivot(graph, nodes[0])

        # Build expected optimal spanning tree.
        expected_nodes : List[model.SpanningTreeNode] = [None] * 11
        for i in range(0, 11):
            expected_nodes[i] = model.SpanningTreeNode(idx=i, children=[])

        expected_nodes[0].children.append(model.SpanningTreeEdge(to=expected_nodes[4], edge_idx=2))
        expected_nodes[0].children.append(model.SpanningTreeEdge(to=expected_nodes[1], edge_idx=0))
        expected_nodes[0].children.append(model.SpanningTreeEdge(to=expected_nodes[2], edge_idx=1))
        expected_nodes[2].children.append(model.SpanningTreeEdge(to=expected_nodes[3], edge_idx=1))
        expected_nodes[3].children.append(model.SpanningTreeEdge(to=expected_nodes[10], edge_idx=1))
        expected_nodes[3].children.append(model.SpanningTreeEdge(to=expected_nodes[7], edge_idx=3))
        expected_nodes[7].children.append(model.SpanningTreeEdge(to=expected_nodes[8], edge_idx=0))
        expected_nodes[7].children.append(model.SpanningTreeEdge(to=expected_nodes[6], edge_idx=2))
        expected_nodes[6].children.append(model.SpanningTreeEdge(to=expected_nodes[5], edge_idx=0))
        expected_nodes[5].children.append(model.SpanningTreeEdge(to=expected_nodes[9], edge_idx=1))

        assert_spanning_trees(self, state.root, expected_nodes[0])

    def test_primal_pivot_on_unbounded_graph(self) -> None:
        # See https://youtu.be/zgtY5nGAMgY?t=3147.
        # Build graph.
        graph = model.Graph()

        # Add nodes from `a` to `h`.
        graph.add_node(0, resource=-4) # a
        graph.add_node(1, resource=2) # b
        graph.add_node(2, resource=1) # c
        graph.add_node(3, resource=0) # d
        graph.add_node(4, resource=-4) # e
        graph.add_node(5, resource=5) # f
        graph.add_node(6, resource=-5) # g
        graph.add_node(7, resource=-1) # h
        graph.add_node(8, resource=9) # i

        # Add edges.
        # Bottom triangle.
        graph.add_edge(origin=0, dest=4, cost=-10)
        graph.add_edge(origin=1, dest=0, cost=1)
        graph.add_edge(origin=4, dest=1, cost=8)
        # Rhomboid.
        graph.add_edge(origin=4, dest=3, cost=2)
        graph.add_edge(origin=3, dest=2, cost=4)
        graph.add_edge(origin=2, dest=1, cost=1)
        # Trapezoid.
        graph.add_edge(origin=5, dest=4, cost=2)
        graph.add_edge(origin=5, dest=8, cost=15)
        graph.add_edge(origin=3, dest=8, cost=4)
        # Top triangle.
        graph.add_edge(origin=6, dest=5, cost=7)
        graph.add_edge(origin=7, dest=6, cost=2)
        graph.add_edge(origin=8, dest=7, cost=4)

        root = model.spanning_tree(graph)

        with self.assertRaises(model.PrimalUnboundedError):
            model.primal_pivot(graph, root)


def assert_spanning_trees(
    case: unittest.TestCase, 
    root: model.SpanningTreeNode, 
    other_root: model.SpanningTreeNode,
) -> None:
    case.assertEqual(root.idx, other_root.idx)
    case.assertEqual(len(root.children), len(other_root.children))
    matchedIdxCount = 0

    for i in range(len(root.children)):
        for j in range(len(other_root.children)):
            if root.children[i].to.idx == other_root.children[j].to.idx:
                case.assertEqual(root.children[i].edge_idx, other_root.children[j].edge_idx)
                assert_spanning_trees(case, root.children[i].to, other_root.children[j].to)

                matchedIdxCount += 1
                break

    case.assertEqual(matchedIdxCount, len(root.children))


def bipartite_unidirectional_graph() -> model.Graph:
    graph = model.Graph()

    graph.add_node(0, 1)
    graph.add_node(1, 2)
    graph.add_node(2, 5)

    graph.add_node(3, -6)
    graph.add_node(4, -2)

    graph.add_edge(0, 3, cost=0)
    graph.add_edge(0, 4, cost=1)
    graph.add_edge(1, 3, cost=5.5)
    graph.add_edge(1, 4, cost=6)
    graph.add_edge(2, 3, cost=0)
    graph.add_edge(2, 4, cost=1)

    return graph


def infeasible_balanced_graph() -> model.Graph:
    graph = model.Graph()

    graph.add_node(0, resource=1)
    graph.add_node(1, resource=-3)
    graph.add_node(2, resource=-3)
    graph.add_node(3, resource=-5)
    graph.add_node(4, resource=7)
    graph.add_node(5, resource=3)

    # Weights don't matter in flow assignment.
    graph.add_edge(2, 0, cost=1)
    graph.add_edge(2, 1, cost=5)
    graph.add_edge(2, 3, cost=2)
    graph.add_edge(4, 3, cost=8)
    graph.add_edge(5, 3, cost=1)

    return graph


def infeasible_graph_full_spanning_tree() -> model.SpanningTreeNode:
    # Build spanning tree.
    node0 = model.SpanningTreeNode(0, children=[])
    node1 = model.SpanningTreeNode(1, children=[])
    node2 = model.SpanningTreeNode(2, children=[])
    node3 = model.SpanningTreeNode(3, children=[])
    node4 = model.SpanningTreeNode(4, children=[])
    node5 = model.SpanningTreeNode(5, children=[])

    node5.children.append(model.SpanningTreeEdge(to=node3, edge_idx=0))
    node3.children.append(model.SpanningTreeEdge(to=node4, edge_idx=1))
    node3.children.append(model.SpanningTreeEdge(to=node2, edge_idx=0))
    node2.children.append(model.SpanningTreeEdge(to=node0, edge_idx=0))
    node2.children.append(model.SpanningTreeEdge(to=node1, edge_idx=1))

    return node5


def letter_shape_graph() -> model.Graph:
    graph = model.Graph()

    graph.add_node(0, resource=4)
    graph.add_node(1, resource=0)
    graph.add_node(2, resource=-2)
    graph.add_node(3, resource=3)
    graph.add_node(4, resource=-8)
    graph.add_node(5, resource=3)

    graph.add_edge(origin=0, dest=1, cost=2)
    graph.add_edge(origin=0, dest=2, cost=3)
    graph.add_edge(origin=1, dest=3, cost=7)
    graph.add_edge(origin=2, dest=3, cost=1)
    graph.add_edge(origin=3, dest=4, cost=2)
    graph.add_edge(origin=5, dest=3, cost=4)
    graph.add_edge(origin=4, dest=1, cost=7)
    graph.add_edge(origin=2, dest=5, cost=9)
    graph.add_edge(origin=5, dest=4, cost=2)

    return graph
