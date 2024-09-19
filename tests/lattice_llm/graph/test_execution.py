from dataclasses import dataclass
from typing import Sequence

from mypy_boto3_bedrock_runtime.type_defs import MessageOutputTypeDef
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message

from lattice_llm.bedrock import ModelId, converse, BedrockClient, FakeBedrockClient, FakeBedrockModel
from lattice_llm.bedrock.messages import text
from lattice_llm.graph import END, Graph, GraphExecutionResult
from lattice_llm.graph import run_graph
from lattice_llm.state import LocalStateStore


@dataclass
class Context:
    user_id: str
    bedrock: BedrockClient


@dataclass
class State:
    messages: list[Message]


class FakeClaude(FakeBedrockModel):
    id = ModelId.CLAUDE_3_5

    def generate_response(self, messages: Sequence[MessageUnionTypeDef]) -> MessageOutputTypeDef:
        return {"role": "assistant", "content": [{"text": "I'm Claude, a helpful AI assistant!"}]}


def welcome(context: Context, state: State) -> State:
    return State(messages=state.messages + [text("Hello!", role="assistant")])


def assistant(context: Context, state: State) -> State:
    response = converse(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        prompt="You are a helpful assistant.",
    )

    message = response["output"]["message"]
    return State(messages=state.messages + [message])


def test_run_graph() -> None:
    context = Context("user-1", bedrock=FakeBedrockClient([FakeClaude()]))
    store = LocalStateStore(lambda: State(messages=[]))
    graph = Graph[Context, State](nodes=[welcome, assistant], edges=[(welcome, assistant), (assistant, END)])

    expected_messages = [
        text("Hello!", role="assistant"),
        text("I'm Claude, a helpful AI assistant!", role="assistant"),
    ]

    results = [result for result in run_graph(graph, context, store, context.user_id)]
    assert results == [
        GraphExecutionResult(
            state=State(messages=expected_messages[0:1]),
            nodes_executed=[welcome.__name__],
            is_finished=False,
        ),
        GraphExecutionResult(
            state=State(messages=expected_messages),
            nodes_executed=[assistant.__name__],
            is_finished=False,
        ),
        GraphExecutionResult(
            state=State(messages=expected_messages),
            nodes_executed=[END],
            is_finished=True,
        ),
    ]


def test_run_graph_with_user_input() -> None:
    context = Context("user-2", bedrock=FakeBedrockClient([FakeClaude()]))
    store = LocalStateStore(lambda: State(messages=[]))
    graph = Graph[Context, State](nodes=[welcome, assistant], edges=[(welcome, assistant), (assistant, END)])

    welcome_response = text("Hello!", role="assistant")
    assistant_response = text("I'm Claude, a helpful AI assistant!", role="assistant")
    user_response = text("<user response>")

    results: list[GraphExecutionResult[State]] = []
    for result in run_graph(graph, context, store, context.user_id):
        store.set(context.user_id, State(messages=result.state.messages + [user_response]))
        results.append(result)

    assert results == [
        GraphExecutionResult(
            state=State(messages=[welcome_response]),
            nodes_executed=[welcome.__name__],
            is_finished=False,
        ),
        GraphExecutionResult(
            state=State(messages=[welcome_response, user_response, assistant_response]),
            nodes_executed=[assistant.__name__],
            is_finished=False,
        ),
        GraphExecutionResult(
            state=State(messages=[welcome_response, user_response, assistant_response, user_response]),
            nodes_executed=[END],
            is_finished=True,
        ),
    ]


def test_execute_graph_at_specified_node() -> None:
    context = Context("user-1", bedrock=FakeBedrockClient([FakeClaude()]))
    store = LocalStateStore(lambda: State(messages=[]))
    graph = Graph[Context, State](nodes=[welcome, assistant], edges=[(welcome, assistant), (assistant, END)])

    expected_messages = [
        text("I'm Claude, a helpful AI assistant!", role="assistant"),
    ]

    result = graph.execute(context, store.get(context.user_id), from_node=[welcome.__name__])

    assert result == GraphExecutionResult(
        state=State(messages=expected_messages),
        nodes_executed=[assistant.__name__],
        is_finished=False,
    )
