from dataclasses import dataclass
from typing import Callable, Optional, Self

import boto3
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message
from pydantic import BaseModel

from lattice_llm.bedrock import BedrockClient, ModelId, converse, converse_with_structured_output, text
from lattice_llm.graph import END, Graph, run_chatbot_on_cli
from lattice_llm.state import LocalStateStore

from .prompts import act_1_prompt, act_2_prompt, act_3_prompt, character_creation_prompt, end_game_prompt


@dataclass
class Context:
    """Context that a Graph can utilize as it executes. Context is not intended to be mutated"""

    user_id: str
    bedrock: BedrockClient
    tools: list[Callable]


class PlayerCharacter(BaseModel):
    name: str
    """The character's name."""

    character_class: str
    """The character's class"""

    level: int
    """The character's current level"""


@dataclass
class State:
    """State that a Graph can update as it executes."""

    messages: list[Message]
    character: Optional[PlayerCharacter] = None

    @classmethod
    def merge(cls, a: Self, b: Self) -> Self:
        return cls(messages=a.messages + b.messages)


class GameState(BaseModel):
    character_creation_complete: bool = False
    """True if the user has created a character and is ready to begin the game. False otherwise."""

    act_1_complete: bool = False
    """True if Act 1 of the adventure has been completed. False if not."""

    act_2_complete: bool = False
    """True if Act 2 of the adventure has been completed. False if not."""

    act_3_complete: bool = False
    """True if Act 3 of the adventure has been completed. False if not."""


def character_creation(context: Context, state: State) -> State:
    response = converse(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        prompt=character_creation_prompt(),
    )

    message = response["output"]["message"]
    return State.merge(state, State(messages=[message]))


def act_1(context: Context, state: State) -> State:
    response = converse(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        prompt=act_1_prompt(),
    )

    message = response["output"]["message"]
    return State.merge(state, State(messages=[message]))


def act_2(context: Context, state: State) -> State:
    response = converse(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        prompt=act_2_prompt(),
    )

    message = response["output"]["message"]
    return State.merge(state, State(messages=[message]))


def act_3(context: Context, state: State) -> State:
    response = converse(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        prompt=act_3_prompt(),
    )

    message = response["output"]["message"]
    return State.merge(state, State(messages=[message]))


def end_game(context: Context, state: State) -> State:
    response = converse(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        prompt=end_game_prompt(),
    )

    message = response["output"]["message"]
    return State.merge(state, State(messages=[message]))


def continue_or_end(context: Context, state: State) -> str:
    response = converse_with_structured_output(
        client=context.bedrock,
        model_id=ModelId.CLAUDE_3_5,
        messages=state.messages,
        prompt="Extract the current state of the game from the previous messages.",
        output_schema=GameState,
    )

    if not response.character_creation_complete:
        return character_creation.__name__
    elif response.character_creation_complete and not response.act_1_complete:
        return act_1.__name__
    elif not response.act_2_complete:
        return act_2.__name__
    elif not response.act_3_complete:
        return act_3.__name__
    else:
        return END


context = Context(bedrock=boto3.client("bedrock-runtime"), user_id="user-1", tools=[])
graph = Graph[Context, State](
    nodes=[character_creation, act_1, act_2, act_3, end_game],
    edges=[
        (character_creation, continue_or_end),
        (act_1, continue_or_end),
        (act_2, continue_or_end),
        (act_3, continue_or_end),
        (end_game, END),
    ],
)

store = LocalStateStore(lambda: State(messages=[text("...")]))

run_chatbot_on_cli(graph, context, store)
