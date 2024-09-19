from dataclasses import dataclass, field, replace
from typing import Optional, Self

from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message

from lattice_llm.bedrock import text
from lattice_llm.graph import END, START, Graph, GraphExecutionResult


@dataclass
class Context:
    user_id: str = "user-1"


@dataclass
class State:
    messages: list[Message] = field(default_factory=list)

    def append_message(self, message: Message) -> Self:
        return replace(self, messages=self.messages + [message])


def execute_graph(
    graph: Graph[Context, State], context: Context = Context(), state: State = State(), depth: Optional[int] = None
) -> list[GraphExecutionResult]:
    current_state = state
    is_finished = False
    results: list[GraphExecutionResult] = []
    iteration: int = 1
    while is_finished == False:
        from_node = results[-1].nodes_executed if len(results) > 0 else [START]
        result = graph.execute(context, current_state, from_node=from_node)
        current_state = result.state
        is_finished = result.is_finished
        results.append(result)

        if depth != None and depth == iteration:
            break

        iteration += 1

    return results


welcome_msg = text("Welcome", role="assistant")
goodbye_msg = text("Goodbye", role="assistant")


def welcome(context: Context, state: State) -> State:
    return state.append_message(welcome_msg)


def goodbye(context: Context, state: State) -> State:
    return state.append_message(goodbye_msg)


def test_single_node_graph_executes() -> None:
    graph = Graph[Context, State](nodes=[welcome])
    result = graph.execute(Context(), State())

    assert result == GraphExecutionResult(
        state=State([text("Welcome", role="assistant")]),
        is_finished=False,
        nodes_executed=[welcome.__name__],
    )


def test_single_node_graph_terminates() -> None:
    graph = Graph[Context, State](nodes=[welcome])
    [_, result_2] = execute_graph(graph)

    assert result_2 == GraphExecutionResult(
        state=State([welcome_msg]),
        is_finished=True,
        nodes_executed=[END],
    )


def test_edge_is_followed() -> None:
    graph = Graph[Context, State](nodes=[welcome, goodbye])
    graph.add_edge(welcome, goodbye)

    [_, result_2, result_3] = execute_graph(graph, depth=3)

    assert result_2 == GraphExecutionResult(
        state=State([welcome_msg, goodbye_msg]),
        is_finished=False,
        nodes_executed=[goodbye.__name__],
    )

    assert result_3 == GraphExecutionResult(
        state=State([welcome_msg, goodbye_msg]),
        is_finished=True,
        nodes_executed=[END],
    )


def test_cycles_are_ok() -> None:
    graph = Graph[Context, State](
        nodes=[welcome, goodbye],
        edges=[
            (welcome, goodbye),
            (goodbye, welcome),
        ],
    )

    [_, __, result_3, result_4] = execute_graph(graph, depth=4)

    assert result_3 == GraphExecutionResult(
        state=State([welcome_msg, goodbye_msg, welcome_msg]),
        is_finished=False,
        nodes_executed=[welcome.__name__],
    )

    assert result_4 == GraphExecutionResult(
        state=State([welcome_msg, goodbye_msg, welcome_msg, goodbye_msg]),
        is_finished=False,
        nodes_executed=[goodbye.__name__],
    )


def test_graph_with_explicit_ids_executes() -> None:
    graph = Graph[Context, State](
        nodes=[("welcome", welcome), ("goodbye", goodbye)],
        edges=[
            ("welcome", goodbye),
        ],
    )

    [_, __, result_3] = execute_graph(graph)

    assert result_3 == GraphExecutionResult(
        state=State([welcome_msg, goodbye_msg]),
        is_finished=True,
        nodes_executed=[END],
    )


def test_conditional_edge_is_followed() -> None:

    def say_one(context: Context, state: State) -> State:
        return state.append_message(text("One", role="assistant"))

    def say_two(context: Context, state: State) -> State:
        return state.append_message(text("Two", role="assistant"))

    def conditional_edge(context: Context, state: State) -> str:
        if context.user_id == "user-1":
            return say_one.__name__
        elif context.user_id == "user-2":
            return say_two.__name__
        else:
            return END

    graph = Graph[Context, State](nodes=[welcome, say_one, say_two])
    graph.add_edge(welcome, conditional_edge)

    [_, say_one_result] = execute_graph(graph, context=Context(user_id="user-1"), depth=2)
    assert say_one_result == GraphExecutionResult(
        state=State([welcome_msg, text("One", role="assistant")]),
        is_finished=False,
        nodes_executed=[say_one.__name__],
    )

    [_, say_two_result] = execute_graph(graph, context=Context(user_id="user-2"), depth=2)
    assert say_two_result == GraphExecutionResult(
        state=State([welcome_msg, text("Two", role="assistant")]),
        is_finished=False,
        nodes_executed=[say_two.__name__],
    )
