import ast
import inspect
from inspect import getsource


from lattice_llm.graph import Graph
from lattice_llm.graph.execution import ChatbotContext, ChatbotState
from .models import Node, Edge, Message, TextContentBlock


def _get_return_ids(f) -> list[str]:
    """Does some nasty AST hackery to figure out which node(s) a conditional edge returns"""

    def get_ids(elt) -> list[str]:
        if isinstance(elt, (ast.Tuple,)):
            return [str(x.id) for x in elt.elts if isinstance(x, (ast.Name,))]
        elif isinstance(elt, (ast.Name,)):
            return [str(elt.id)]
        elif isinstance(elt, ast.Constant):
            return [str(elt.value)]
        elif isinstance(elt, ast.IfExp):
            return [str(elt.body.value), str(elt.orelse.value)]
        else:
            return []

    (tree,) = ast.parse(inspect.getsource(f)).body
    returns = set[str]()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Return,)):
            for r in get_ids(node.value):
                returns.add(r)

    return list(returns)


def map_nodes(graph: Graph[ChatbotContext, ChatbotState], last_nodes_executed: list[str]) -> list[Node]:
    return [
        Node(id=id, source=getsource(node), is_active=(id in last_nodes_executed)) for id, node in graph.nodes.items()
    ]


def map_edges(graph: Graph[ChatbotContext, ChatbotState]) -> list[Edge]:
    edges: list[Edge] = []

    for source_id, destinations in graph.edges.items():
        for destination in destinations:
            match destination:
                case str():
                    edges.append(Edge(source_id=source_id, destination_id=destination))
                case destination if callable(destination):
                    if graph.nodes.get(destination.__name__):
                        destination_id = destination.__name__
                        edges.append(Edge(source_id=source_id, destination_id=destination_id))
                    else:
                        destination_ids = _get_return_ids(destination)
                        for destination_id in destination_ids:
                            edges.append(Edge(source_id=source_id, destination_id=destination_id))
    return edges


def map_messages(state: ChatbotState) -> list[Message]:
    messages: list[Message] = []
    for message in state.messages:
        text_blocks = [TextContentBlock(text=block["text"]) for block in message["content"] if block.get("text")]
        messages.append(Message(role=message["role"], content=text_blocks))

    return messages
