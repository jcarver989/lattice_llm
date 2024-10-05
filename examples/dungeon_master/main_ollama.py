from dataclasses import dataclass
from typing import Callable, Generator, Optional, Self

from mypy_boto3_bedrock_runtime.type_defs import ConverseOutputTypeDef
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message

from examples.dungeon_master.prompts import (
    act_1_prompt,
    act_2_prompt,
    act_3_prompt,
    character_creation_prompt,
    end_game_prompt,
    _BASE_PERSONA,
)
from lattice_llm.bedrock.messages import text
from lattice_llm.graph import END, Graph, Node
from lattice_llm.graph.execution import run_chatbot_on_cli
from lattice_llm.ollama.converse import ModelId, converse_streaming, converse_with_structured_output
from lattice_llm.state import LocalStateStore
from .player_character import PlayerCharacter, AbilityScores, NameAndCharacterClass, InventoryItems
from random import randrange


@dataclass
class Context:
    """Context that a Graph can utilize as it executes. Context is not intended to be mutated"""

    user_id: str
    tools: list[Callable]


@dataclass
class State:
    """State that a Graph can update as it executes."""

    messages: list[Message]

    ability_scores: Optional[AbilityScores] = None
    """Random Ability scores rolled during character creation"""

    character: Optional[PlayerCharacter] = None
    """The player character"""

    act_1_complete: bool = False
    """True if Act 1 of the adventure has been completed. False if not."""

    act_2_complete: bool = False
    """True if Act 2 of the adventure has been completed. False if not."""

    act_3_complete: bool = False
    """True if Act 3 of the adventure has been completed. False if not."""

    @classmethod
    def merge(cls, a: Self, b: Self) -> Self:
        return cls(
            messages=a.messages + b.messages,
            ability_scores=a.ability_scores or b.ability_scores,
            character=a.character or b.character,
            act_1_complete=a.act_1_complete or b.act_1_complete,
            act_2_complete=a.act_2_complete or b.act_2_complete,
            act_3_complete=a.act_3_complete or b.act_3_complete,
        )


def process_streaming_response(response_stream: Generator[str, None, None]):
    response: ConverseOutputTypeDef = {"message": {"role": "assistant", "content": []}}

    text_buffer = ""

    for chunk in response_stream:
        text_buffer += chunk
        trimmed_chunk = chunk.rstrip()
        print(chunk, end="", flush=True)

        if trimmed_chunk != "" and trimmed_chunk[-1] in (".", "!", "?", ";"):
            response["message"]["content"].append({"text": text_buffer})
            # subprocess.run(["say", text_buffer])
            text_buffer = ""

    if len(text_buffer) > 0:
        response["message"]["content"].append({"text": text_buffer})
        # subprocess.run(["say", text_buffer])

    return response


def character_creation_intro(context: Context, state: State) -> State:
    ability_scores = AbilityScores.get_random_scores()

    response_stream = converse_streaming(
        model_id=ModelId.LLAMA_3_1,
        prompt=character_creation_prompt(ability_scores),
        messages=state.messages,
    )

    response = process_streaming_response(response_stream)
    return State.merge(state, State(messages=[response["message"]], ability_scores=ability_scores))


def create_character(context: Context, state: State) -> State:

    name_and_class = converse_with_structured_output(
        model_id=ModelId.LLAMA_3_1,
        messages=state.messages,
        output_schema=NameAndCharacterClass,
        prompt=f"""
        {_BASE_PERSONA}

        Extract the user's chosen name and character class and return it as JSON, using the following JSON schema:

        Schema
        ------
        {NameAndCharacterClass.model_json_schema()}
        ------

        Respond only in valid JSON.
        """,
    )

    inventory_items = converse_with_structured_output(
        model_id=ModelId.LLAMA_3_1,
        messages=state.messages,
        output_schema=InventoryItems,
        prompt=f"""
        {_BASE_PERSONA}

        Generate 3 inventory items for the player character. These items should be thematic to the player's chosen class. Output these items as JSON using the following JSON schema:
        
        # BEGIN JSON SCHEMA
        {InventoryItems.model_json_schema()}
        # END JSON SCHEMA

        Respond only in valid JSON.
        """,
    )

    character = PlayerCharacter(
        name=name_and_class.name,
        abillity_scores=state.ability_scores,
        character_class=name_and_class.character_class,
        iventory_items=inventory_items.items,
        level=1,
        hp=randrange(1, 6),
    )

    response_stream = converse_streaming(
        model_id=ModelId.LLAMA_3_1,
        prompt=f"""
        {_BASE_PERSONA}

        The player has just finished generating a character for the game. Their character is represented as JSON below:

        # BEGIN JSON
        {character.model_dump_json()}
        # END JSON

        Tell the player about their character and what's in their inventory. Don't respond with JSON. Stop after you've described their character, do not begin the adventure yet.  
        """,
        messages=state.messages,
    )

    response = process_streaming_response(response_stream)

    return State.merge(state, State(messages=state.messages + [response["message"]], character=character))


def act_1(context: Context, state: State) -> State:
    response_stream = converse_streaming(
        model_id=ModelId.LLAMA_3_1,
        prompt=act_1_prompt(),
        messages=state.messages,
    )

    response = process_streaming_response(response_stream)
    return State.merge(state, State(messages=[response["message"]]))


def act_2(context: Context, state: State) -> State:
    response_stream = converse_streaming(
        model_id=ModelId.LLAMA_3_1,
        prompt=act_2_prompt(),
        messages=state.messages,
    )

    response = process_streaming_response(response_stream)
    return State.merge(state, State(messages=[response["message"]]))


def act_3(context: Context, state: State) -> State:
    response_stream = converse_streaming(
        model_id=ModelId.LLAMA_3_1,
        prompt=act_3_prompt(),
        messages=state.messages,
    )

    response = process_streaming_response(response_stream)
    return State.merge(state, State(messages=[response["message"]]))


def end_game(context: Context, state: State) -> State:
    response_stream = converse_streaming(
        model_id=ModelId.LLAMA_3_1,
        prompt=end_game_prompt(),
        messages=state.messages,
    )

    response = process_streaming_response(response_stream)
    return State.merge(state, State(messages=[response["message"]]))


def continue_or_end(context: Context, state: State) -> Node[Context, State]:

    if not state.character:
        return character_creation_intro
    elif not state.act_1_complete:
        print("off to act 1!")
        return act_1
    elif not state.act_2_complete:
        print("off to act 2!")
        return act_2
    else:
        print("off to act 3!")
        return act_3


context = Context(user_id="user-1", tools=[])
graph = Graph[Context, State](
    nodes=[character_creation_intro, create_character, act_1, act_2, act_3, end_game],
    edges=[
        (character_creation_intro, create_character),
        (create_character, continue_or_end),
        (act_1, continue_or_end),
        (act_2, continue_or_end),
        (act_3, continue_or_end),
        (end_game, END),
    ],
)

store = LocalStateStore(lambda: State(messages=[text("...")]))


# run_graph_on_streamlit(graph, context, State(messages=[text("...")]))

run_chatbot_on_cli(graph, context, store)
