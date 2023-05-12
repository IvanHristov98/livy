import unittest

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

    def test_supply_provision_along_strong_edge(self) -> None:
        node3 = model.SpanningTreeNode(idx=3, children=[])
        self._root.children.append(model.SpanningTreeEdge(to=node3, edge_idx=0))

        flows = model.assign_flow_values(self._graph, self._root)

        self.assertEqual(len(flows), 1)
        self.assertIn(0, flows)
        self.assertIn(model.FlowEdge(dest=3, val=6), flows[0])


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