from dataclasses import dataclass
from typing import Callable, Self

import boto3
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message
from pydantic import BaseModel

from lattice_llm.bedrock import BedrockClient, ModelId, converse, converse_with_structured_output
from lattice_llm.bedrock.messages import text
from lattice_llm.graph import END, Graph, Node
from lattice_llm.state import LocalStateStore
from lattice_llm.streamlit.run_graph import run_graph_on_streamlit


@dataclass
class Context:
    """Context that a Graph can utilize as it executes. Context is not intended to be mutated"""

    user_id: str
    bedrock: BedrockClient
    tools: list[Callable]


@dataclass
class State:
    """State that a Graph can update as it executes."""

    messages: list[Message]

    @classmethod
    def merge(cls, a: Self, b: Self) -> Self:
        return cls(messages=a.messages + b.messages)


class ConversationDetails(BaseModel):
    should_continue: bool = True
    """True if the user wishes to keep conversing. False if the user has indicated a desire to end the conversation. If ambiguous, assume the user wants to continue the conversation."""


def welcome(context: Context, state: State) -> State:
    """A graph node that returns a fixed (canned) response."""
    return State.merge(state, State(messages=[text("...", role="user"), text("Hello!", role="assistant")]))


def assistant(context: Context, state: State) -> State:
    """A graph node that returns a message from Claude 3.5 Sonnet via the boto3 Bedrock client"""
    response = converse(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        tools=context.tools,
        prompt="You are a helpful assistant.",
    )

    message = response["output"]["message"]
    return State.merge(state, State(messages=[message]))


def goodbye(context: Context, state: State) -> State:
    """A graph node that returns another fixed (canned) response to say goodbye to the user."""
    return State.merge(state, State(messages=[text("Goodbye!", role="assistant")]))


def continue_or_end(context: Context, state: State) -> Node[Context, State]:
    """A conditional edge, extracts structured output from Claude, in the form of a ConversationDetails Pydantic model and uses it to determine if we should loop back to the assistant node, or proceed to the goodbye node."""
    response = converse_with_structured_output(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        prompt="Extract the conversation details from historical messages.",
        output_schema=ConversationDetails,
    )

    if response.should_continue:
        return assistant
    else:
        return goodbye


def get_temperature(city: str) -> int:
    """
    Returns the current temperature for a city.

    :param city: The city to pull temperature information from
    :return: The temperature in degrees fahrenheit for the specified city.
    """

    return 50


context = Context(bedrock=boto3.client("bedrock-runtime"), user_id="user-1", tools=[get_temperature])
graph = Graph[Context, State](
    nodes=[welcome, assistant, goodbye],
    edges=[
        (welcome, assistant),
        (assistant, continue_or_end),
        (goodbye, END),
    ],
)

store = LocalStateStore(lambda: State(messages=[]))


run_graph_on_streamlit(graph, context, State(messages=[]))
