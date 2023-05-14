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

        self._assert_spanning_trees(root, expected_root)

    def _assert_spanning_trees(self, root: model.SpanningTreeNode, other_root: model.SpanningTreeNode) -> None:
        self.assertEqual(root.idx, other_root.idx)
        self.assertEqual(len(root.children), len(other_root.children))

        for i in range(len(root.children)):
            self.assertEqual(root.children[i].edge_idx, other_root.children[i].edge_idx)
            self._assert_spanning_trees(root.children[i].to, other_root.children[i].to)


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
