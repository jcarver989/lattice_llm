from dataclasses import dataclass
from typing import Callable, Optional, Self

from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message
from pydantic import BaseModel

from examples.dungeon_master.prompts import (
    act_1_prompt,
    act_2_prompt,
    act_3_prompt,
    character_creation_prompt,
    end_game_prompt,
)
from lattice_llm.bedrock.messages import text
from lattice_llm.graph import END, Graph, Node
from lattice_llm.ollama.converse import ModelId, converse, converse_with_structured_output
from lattice_llm.state import LocalStateStore


@dataclass
class Context:
    """Context that a Graph can utilize as it executes. Context is not intended to be mutated"""

    user_id: str
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
        model_id=ModelId.LLAMA_3_1,
        prompt=character_creation_prompt(),
        messages=state.messages,
    )

    return State.merge(state, State(messages=[response["message"]]))


def act_1(context: Context, state: State) -> State:
    response = converse(
        model_id=ModelId.LLAMA_3_1,
        prompt=act_1_prompt(),
        messages=state.messages,
    )

    return State.merge(state, State(messages=[response["message"]]))


def act_2(context: Context, state: State) -> State:
    response = converse(
        model_id=ModelId.LLAMA_3_1,
        prompt=act_2_prompt(),
        messages=state.messages,
    )

    return State.merge(state, State(messages=[response["message"]]))


def act_3(context: Context, state: State) -> State:
    response = converse(
        model_id=ModelId.LLAMA_3_1,
        prompt=act_3_prompt(),
        messages=state.messages,
    )

    return State.merge(state, State(messages=[response["message"]]))


def end_game(context: Context, state: State) -> State:
    response = converse(
        model_id=ModelId.LLAMA_3_1,
        prompt=end_game_prompt(),
        messages=state.messages,
    )

    return State.merge(state, State(messages=[response["message"]]))


def continue_or_end(context: Context, state: State) -> Node[Context, State]:
    game_state = converse_with_structured_output(
        model_id=ModelId.LLAMA_3_1, messages=state.messages, output_schema=GameState
    )

    if not game_state.character_creation_complete:
        return character_creation
    elif game_state.character_creation_complete and not game_state.act_1_complete:
        return act_1
    elif not game_state.act_2_complete:
        return act_2
    else:
        return act_3


context = Context(user_id="user-1", tools=[])
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

from lattice_llm.streamlit.run_graph import run_graph_on_streamlit

run_graph_on_streamlit(graph, context, State(messages=[text("...")]))

# run_chatbot_on_cli(graph, context, store)
