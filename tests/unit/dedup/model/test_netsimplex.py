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
        self._graph = foo_graph()

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
        self._graph = foo_graph()
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
        graph.add_edge(2, 3, cost=1)
        graph.add_edge(4, 3, cost=8)
        graph.add_edge(5, 4, cost=1)

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

        flows = model.assign_flow_values(graph, node5)

        self.assertIn(2, flows)
        self.assertIn(model.FlowEdge(dest=0, val=-1), flows[2])
        self.assertIn(model.FlowEdge(dest=1, val=3), flows[2])

        self.assertIn(3, flows)
        self.assertIn(model.FlowEdge(dest=2, val=-5), flows[3])
        self.assertIn(model.FlowEdge(dest=4, val=7), flows[3])

        self.assertIn(5, flows)
        self.assertIn(model.FlowEdge(dest=3, val=3), flows[5])


def foo_graph() -> model.Graph:
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