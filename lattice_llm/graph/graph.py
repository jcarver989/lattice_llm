from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, cast

ID = str
START = "start"
END = "end"


T = TypeVar("T")
U = TypeVar("U")

Node = Callable[[T, U], Optional[U]]
NodeOrId = ID | Node[T, U]
NodeOrNodeWithId = Node[T, U] | tuple[ID, Node[T, U]]

ConditionalEdgeDestination = Callable[[T, U], Optional[NodeOrId[T, U]]]
EdgeDestination = NodeOrId[T, U] | ConditionalEdgeDestination[T, U]
Middleware = Callable[[ID, T], None]


@dataclass
class GraphExecutionResult(Generic[U]):
    state: U
    nodes_executed: list[ID]
    is_finished: bool


class Graph(Generic[T, U]):
    """An immutable Graph. Graphs are executed in a breadth-first fashion."""

    context: T
    root_node: ID
    nodes: dict[ID, Node[T, U]]
    edges: dict[ID, list[EdgeDestination[T, U]]]
    middleware: list[Middleware[U]]

    def __init__(
        self,
        nodes: Optional[list[NodeOrNodeWithId[T, U]]] = None,
        edges: Optional[list[tuple[NodeOrId[T, U], EdgeDestination[T, U]]]] = None,
        middleware: list[Middleware[U]] = [],
    ):
        self.nodes = {}
        self.edges = {}
        self.middleware = middleware

        if nodes:
            for i, n in enumerate(nodes):
                is_root = True if i == 0 else None
                match n:
                    case node if callable(node):
                        self.add_node(node=node, is_root=is_root)
                    case (id, node):
                        self.add_node(node=node, id=id, is_root=is_root)
        if edges:
            for source, destination in edges:
                self.add_edge(source, destination)

    def add_node(self, node: Node[T, U], id: Optional[ID] = None, is_root: Optional[bool] = None) -> None:
        node_id = id if id else node.__name__
        self.nodes[node_id] = node

        if is_root != None:
            self.root_node = node_id
        elif len(self.nodes) == 1:
            self.root_node = node_id

    def add_edge(self, source: NodeOrId[T, U], destination: EdgeDestination[T, U]) -> None:
        source_id = self._get_node_id(source)
        out_edges = self.edges.setdefault(source_id, [])
        out_edges.append(destination)

    def execute(self, context: T, state: U, from_node: list[ID] = [START]) -> GraphExecutionResult[U]:
        """Executes a single layer in the graph and returns a copy of the updated state."""

        state_copy = deepcopy(state)
        nodes_to_execute = self._get_nodes_to_execute(context, state_copy, from_node)

        if nodes_to_execute == [END]:
            return GraphExecutionResult(
                state=state_copy,
                nodes_executed=nodes_to_execute,
                is_finished=True,
            )

        for node_id in nodes_to_execute:
            node = self.nodes[node_id]
            state_copy = node(context, state_copy) or state_copy

        return GraphExecutionResult(
            state=state_copy,
            nodes_executed=nodes_to_execute,
            is_finished=False,
        )

    def _get_nodes_to_execute(self, context: T, state: U, from_node: list[ID]) -> list[ID]:
        if from_node == [START]:
            return [self.root_node]

        nodes_to_execute: list[ID] = []
        for start_node in from_node:
            for node in self._get_connected_nodes(start_node, context, state):
                nodes_to_execute.append(node)

        return nodes_to_execute if len(nodes_to_execute) > 0 else [END]

    def _get_connected_nodes(self, node: ID, context: T, state: U) -> list[ID]:
        children: list[ID] = []
        for edge in self.edges.setdefault(node, []):
            child_id = self._get_destination_id(context, state, edge)
            if child_id:
                children.append(child_id)

        return children

    def _get_destination_id(self, context: T, state: U, edge_destination: EdgeDestination[T, U]) -> Optional[ID]:
        match edge_destination:
            case str():
                return edge_destination

            case function if callable(function):
                if self.nodes.get(function.__name__):
                    return function.__name__
                else:
                    conditional_edge = cast(ConditionalEdgeDestination[T, U], function)
                    node_or_id = conditional_edge(context, state)
                    if node_or_id:
                        return self._get_node_id(node_or_id)
                    else:
                        return None
            case _:
                raise NotImplementedError()

    def _get_node_id(self, node: NodeOrId[T, U]) -> ID:
        return node.__name__ if callable(node) else node
