from typing import Generator, Protocol, TypeVar, Callable

from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message

from ..bedrock import text, maybe_execute_tools
from ..util import Color, color_text, print_message
from .graph import START, Graph, GraphExecutionResult
from ..state import StateStore


class ChatbotContext(Protocol):
    user_id: str
    tools: list[Callable]


class ChatbotState(Protocol):
    messages: list[Message]


T = TypeVar("T")
U = TypeVar("U")


def run_graph(
    graph: Graph[T, U], context: T, store: StateStore[U], store_key: str
) -> Generator[GraphExecutionResult[U], None, None]:
    """
    Executes a Graph[T, U] via a generator, yielding a GraphExecutionResult and control back to the caller each time a layer is executed. Execution occurs in a breadth-first fashion.
    """
    is_finished = False
    last_nodes_executed = [START]

    while is_finished != True:
        state = store.get(store_key)
        result = graph.execute(context, state, from_node=last_nodes_executed)
        last_nodes_executed = result.nodes_executed
        store.set(store_key, result.state)
        is_finished = result.is_finished
        yield result


V = TypeVar("V", bound=ChatbotContext)
W = TypeVar("W", bound=ChatbotState)


def run_chatbot_on_cli(graph: Graph[V, W], context: V, store: StateStore[W]) -> GraphExecutionResult[W]:
    """
    Runs an interactive 'chatbot' on the command line. 'Chatbot' here is defined as a Graph with context (V) and state (W) that conform to the ChatbotContext and ChatbotState Protocols respectively.
    """
    printed_message_index = 0
    last_result: GraphExecutionResult[W]

    for result in run_graph(graph, context, store, context.user_id):
        while printed_message_index < len(result.state.messages):
            message = result.state.messages[printed_message_index]
            if message["role"] == "assistant":
                print_message(message)

            printed_message_index += 1

        last_message = result.state.messages[-1]
        if last_message["role"] == "assistant":
            tool_results = maybe_execute_tools(last_message, context.tools)
            if tool_results:
                result.state.messages = result.state.messages + [tool_results]
            else:
                user_message = input(f"{color_text('User:', Color.GREEN)} ")
                print("\n")
                result.state.messages = result.state.messages + [text(user_message)]

        last_result = result

    return last_result
