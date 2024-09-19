from typing import Optional
from dataclasses import dataclass
from lattice_llm.bedrock import get_tool_spec, maybe_execute_tools
from pydantic import BaseModel


def test_simple_tool() -> None:
    def sum(a: int, b: int) -> int:
        """Sums two numbers and returns their result"""
        return a + b

    spec = get_tool_spec(sum)
    assert spec == {
        "name": "sum",
        "description": "Sums two numbers and returns their result",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    }


def test_tool_with_optional_param() -> None:
    def sum(a: int, b: Optional[int]) -> int:
        """Sums two numbers and returns their result"""
        return a + (b or 0)

    spec = get_tool_spec(sum)
    assert spec == {
        "name": "sum",
        "description": "Sums two numbers and returns their result",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": ["number", "null"]},
                },
                "required": ["a"],
            },
        },
    }


def test_tool_with_complex_params() -> None:
    @dataclass
    class Vector2:
        a: int
        b: int

    def sum(one: Vector2, two: Vector2) -> Vector2:
        """Sums two Vector2s and returns the result as a Vector2"""
        return Vector2(a=one.a + two.a, b=one.b + two.b)

    spec = get_tool_spec(sum)
    vector2_schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    assert spec == {
        "name": "sum",
        "description": "Sums two Vector2s and returns the result as a Vector2",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "one": vector2_schema,
                    "two": vector2_schema,
                },
                "required": ["one", "two"],
            },
        },
    }


def test_tool_with_union_param() -> None:
    @dataclass
    class Num:
        n: int

    def sum(one: Num | int, two: Num | int) -> int:
        """Sums two numbers and returns the result"""

        match (one, two):
            case (Num(), Num()):
                return one.n + two.n
            case (Num(), int()):
                return one.n + two
            case (int(), Num()):
                return one + two.n
            case (int(), int()):
                return one + two
            case _:
                raise NotImplementedError()

    num_schema = {
        "type": "object",
        "properties": {"n": {"type": "number"}},
        "required": ["n"],
    }

    spec = get_tool_spec(sum)
    assert spec == {
        "name": "sum",
        "description": "Sums two numbers and returns the result",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "one": {"type": [num_schema, "number"]},
                    "two": {"type": [num_schema, "number"]},
                },
                "required": ["one", "two"],
            },
        },
    }


def test_executes_single_tool() -> None:
    def get_temperature(city: str) -> int:
        return 50

    result = maybe_execute_tools(
        tools=[get_temperature],
        message={
            "role": "assistant",
            "content": [
                {"toolUse": {"name": "get_temperature", "input": {"city": "San Francisco"}, "toolUseId": "use-1"}}
            ],
        },
    )

    assert result == {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "use-1", "status": "success", "content": [{"text": "50"}]}}],
    }


def test_executes_multiple_tools() -> None:
    def get_temperature(city: str) -> int:
        return 50

    def get_wind(city: str) -> int:
        return 10

    result = maybe_execute_tools(
        tools=[get_temperature, get_wind],
        message={
            "role": "assistant",
            "content": [
                {"toolUse": {"name": "get_temperature", "input": {"city": "San Francisco"}, "toolUseId": "use-1"}},
                {"toolUse": {"name": "get_wind", "input": {"city": "San Francisco"}, "toolUseId": "use-2"}},
            ],
        },
    )

    assert result == {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "use-1", "status": "success", "content": [{"text": "50"}]}},
            {"toolResult": {"toolUseId": "use-2", "status": "success", "content": [{"text": "10"}]}},
        ],
    }


def test_executes_tool_that_returns_dict() -> None:
    def get_weather(city: str) -> dict[str, int]:
        return {"temperature": 50, "wind": 10, "rain": 0}

    result = maybe_execute_tools(
        tools=[get_weather],
        message={
            "role": "assistant",
            "content": [
                {"toolUse": {"name": "get_weather", "input": {"city": "San Francisco"}, "toolUseId": "use-1"}},
            ],
        },
    )

    assert result == {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "use-1",
                    "status": "success",
                    "content": [{"json": {"temperature": 50, "wind": 10, "rain": 0}}],
                }
            },
        ],
    }


def test_executes_tool_that_returns_pydantic_model() -> None:
    class WeatherDetails(BaseModel):
        temperature: int
        wind: int
        rain: int

    def get_weather(city: str) -> WeatherDetails:
        return WeatherDetails(temperature=50, wind=10, rain=0)

    result = maybe_execute_tools(
        tools=[get_weather],
        message={
            "role": "assistant",
            "content": [
                {"toolUse": {"name": "get_weather", "input": {"city": "San Francisco"}, "toolUseId": "use-1"}},
            ],
        },
    )

    assert result == {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "use-1",
                    "status": "success",
                    "content": [{"json": {"temperature": 50, "wind": 10, "rain": 0}}],
                }
            },
        ],
    }


def test_does_not_fail_if_no_tools_specified() -> None:
    def get_temperature(city: str) -> int:
        return 50

    result = maybe_execute_tools(
        tools=[get_temperature],
        message={
            "role": "assistant",
            "content": [{"text": "Hi"}],
        },
    )

    assert result == None
