import ast
import inspect
from typing import Literal

from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode

from ..graph import END, START, EdgeDestination, Graph


def map_nodes(graph: Graph, last_nodes_executed: list[str]) -> list[StreamlitFlowNode]:
    nodes = [_node(id, is_active=id in last_nodes_executed) for id, node in graph.nodes.items()]
    nodes.append(_node(START, node_type="input"))
    nodes.append(_node(END, node_type="output"))
    return nodes


def map_edges(graph: Graph, root_node: str) -> list[StreamlitFlowEdge]:
    edges = [StreamlitFlowEdge(f"{START}-{root_node}", START, root_node)]

    for source_id, destinations in graph.edges.items():
        for destination in destinations:
            for e in _edge(graph, source_id, destination):
                edges.append(e)

    return edges


def _node(
    id: str, is_active: bool = False, node_type: Literal["default", "input", "output"] = "default"
) -> StreamlitFlowNode:
    return StreamlitFlowNode(
        id=id,
        node_type=node_type,
        data={"content": f"""### {id}"""},
        pos=(0, 0),
        source_position="bottom",
        target_position="top",
        style={"backgroundColor": "green" if is_active else "white"},
    )


def _edge(graph: Graph, source_id: str, destination: EdgeDestination) -> list[StreamlitFlowEdge]:
    match destination:
        case str():
            return [StreamlitFlowEdge(f"{source_id}-{destination}", source_id, destination)]
        case destination if callable(destination):
            if graph.nodes.get(destination.__name__):
                destination_id = destination.__name__
                return [StreamlitFlowEdge(f"{source_id}-{destination_id}", source_id, destination_id)]
            else:
                destination_ids = _get_return_ids(destination)
                return [
                    StreamlitFlowEdge(
                        f"{source_id}-{destination_id}",
                        source_id,
                        destination_id,
                        animated=True,
                        label=f"{destination.__name__}",
                    )
                    for destination_id in destination_ids
                ]


def _get_return_ids(f):
    """Does some nasty AST hackery to figure out which node(s) a conditional edge returns"""

    def get_ids(elt) -> list[str]:
        if isinstance(elt, (ast.Tuple,)):
            return [str(x.id) for x in elt.elts if isinstance(x, (ast.Name,))]
        elif isinstance(elt, (ast.Name,)):
            return [str(elt.id)]
        else:
            return []

    (tree,) = ast.parse(inspect.getsource(f)).body
    returns = set[str]()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Return,)):
            for r in get_ids(node.value):
                returns.add(r)

    return returns
