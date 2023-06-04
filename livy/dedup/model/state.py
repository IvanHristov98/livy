from typing import Dict, List

from livy.dedup.model.spanningtree import SpanningTreeNode
from livy.dedup.model.flow import FlowEdge
from livy.dedup.model.dual import SlackEdge


class SimplexState:
    root: SpanningTreeNode
    flow_vars: Dict[int, List[FlowEdge]]
    dual_vars: Dict[int, float]
    slack_vars: Dict[int, List[SlackEdge]]

    def __init__(
            self,
            root: SpanningTreeNode,
            flow_vars: Dict[int, List[FlowEdge]],
            dual_vars: Dict[int, float],
            slack_vars: Dict[int, List[SlackEdge]],
    ) -> None:
        self.root = root
        self.flow_vars = flow_vars
        self.dual_vars = dual_vars
        self.slack_vars = slack_vars
