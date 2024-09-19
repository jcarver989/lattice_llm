# Lattice LLM

A "lightweight" Python library for building LLM-powered agents as executable `Graph`s. A core goals is to provide a good developer UX.

## Key Features

- **Simple abstractions**. Lattice aims to offer a small set of easy to use abstractions: **Graphs**, **Nodes** and **Edges**.

  - **Graphs** are used to orchestrate steps in an LLM-agent's workflow. A `Graph` executes in breadth-first fashion and has access to caller-provided `Context` (e.g. an AWS Bedrock client, the current user's id etc) and `State` (e.g. persisted chat history).

  - **Nodes** are simply Python functions of the form `(context: Context, state: State) -> State`. They're used to make the agent _do_ stuff. Nodes are intended to be "pure" functions that take the current `Context` + `State` as input and return a copy of the updated `State`.

  - **Edges**: Connect nodes together and provide control flow. They come in two flavors:
    - A tuple of the form `(Node, Node)`, for edges that should always be traversed. Or
    - A Python function of the form `(Context, State) -> Node`, for dynamic routing.

- **Control**. Graphs are executed (by the caller) one layer of at a time. This makes it easy to support use-cases that require waiting on user input before executing the next layer of the `Graph` (e.g. a chatbot running on a web-server).

- **Easy to test and introspect**. Execution can be started from any `Node` in the `Graph`. Each time a `Graph` layer is executed, a `GraphExecutionResult` is returned, which contains the updated `State`. This makes it easy to `assert` on the expected `State` after any `Node` is executed in the `Graph`.

- **Convenience**. Lattice provides the following quality of life features "out of the box":
  - **Persistance** Lattice includes a `StateStore` `Protocol` (interface) for persisting graph `State` and a `LocalStateStore` that provides an in-memory implementation.
  - **AWS Bedrock integration**. Support is provided via a `converse` and `converse_with_structured_output` (which returns structured output in the form of a user-provided Pydantic model)
  - **Tools** Lattice can automatically:
    1. Convert Python functions to the JSON schema format LLMs require for defining tools.
    2. Invoke tools (local Python functions) that an LLM requests to use in its responses.

## Installation

`poetry add lattice_llm`

## Usage

```python
from dataclasses import dataclass
from typing import Callable, Self

import boto3
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message
from pydantic import BaseModel

from lattice_llm.bedrock import BedrockClient, ModelId, converse, converse_with_structured_output
from lattice_llm.bedrock.messages import text
from lattice_llm.graph import END, Graph, Node, run_chatbot_on_cli
from lattice_llm.state import LocalStateStore


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

run_chatbot_on_cli(graph, context, store)

```
