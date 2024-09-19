from inspect import getdoc
from typing import Any, Callable, Dict, Literal, Union, get_args, get_origin, get_type_hints, Optional
from types import UnionType
from pydantic import BaseModel

from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message
from mypy_boto3_bedrock_runtime.type_defs import (
    ToolResultBlockTypeDef,
    ToolResultContentBlockTypeDef,
    ToolSpecificationTypeDef,
    ToolTypeDef,
    ToolUseBlockOutputTypeDef,
    ToolUseBlockTypeDef,
)

""" 
Code here  adapted from https://github.com/phidatahq/phidata/blob/0cd1431d3025a7ad458bddd15a51500d35c4273d/phi/utils/json_schema.py#L26
"""

JsonType = Literal["string", "number", "integer", "object", "array", "boolean", "null"]

py_to_json_type: dict[str, JsonType] = {
    "int": "number",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "NoneType": "null",
    "None": "null",
    "object": "object",
    "list": "array",
}


def is_optional_param(param_json_schema: dict[str, Any]) -> bool:
    param_type = param_json_schema["type"]
    match param_type:
        case list():
            return "null" in param_type
        case str():
            return param_type == "null"
        case _:
            return True


def get_json_schema_for_arg(t: Any) -> dict[str, Any]:
    type_args = get_args(t)
    type_origin = get_origin(t)

    match type_origin:
        case list():
            return {"type": "array", "items": get_json_schema_for_arg(type_args[0])}
        case dict():
            return {"type": "object", "properties": {}}
        case None if py_to_json_type.get(t.__name__):
            return {"type": py_to_json_type.get(t.__name__)}
        case None:
            return get_json_schema_from_type_hints(get_type_hints(t))
        case _ if type_origin == Union or type_origin == UnionType:
            return {
                "type": [
                    py_to_json_type.get(arg.__name__) or get_json_schema_from_type_hints(get_type_hints(arg))
                    for arg in type_args
                ]
            }
        case _:
            return {"type": py_to_json_type.get(t.__name__) or get_json_schema_from_type_hints(get_type_hints(t))}


def get_json_schema_from_type_hints(type_hints: Dict[str, Any]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    for k, v in type_hints.items():
        if k == "return":
            continue
        arg_json_schema = get_json_schema_for_arg(v)
        schema["properties"][k] = arg_json_schema
        if not is_optional_param(arg_json_schema):
            schema["required"].append(k)
    return schema


def get_tool_spec(f: Callable) -> ToolSpecificationTypeDef:
    type_hints = get_type_hints(f)
    return {
        "name": f.__name__,
        "description": getdoc(f) or "",
        "inputSchema": {"json": get_json_schema_from_type_hints(type_hints)},
    }


def get_tool_defs(tools: list[Callable]) -> list[ToolTypeDef]:
    return [{"toolSpec": get_tool_spec(tool)} for tool in tools]


def maybe_execute_tools(message: Message, tools: list[Callable]) -> Optional[Message]:
    name_to_tool: dict[str, Callable] = {}
    for tool in tools:
        name_to_tool[tool.__name__] = tool

    tool_results = [
        execute_tool(block["toolUse"], name_to_tool) for block in message["content"] if block.get("toolUse")
    ]

    if len(tool_results) > 0:
        return {"role": "user", "content": [{"toolResult": tool_result} for tool_result in tool_results]}

    return None


def execute_tool(
    tool_use: ToolUseBlockTypeDef | ToolUseBlockOutputTypeDef, name_to_tool: dict[str, Callable]
) -> ToolResultBlockTypeDef:
    try:
        tool = name_to_tool[tool_use["name"]]
        tool_result = tool(**tool_use["input"])
        return {
            "toolUseId": tool_use["toolUseId"],
            "content": [tool_result_content_block(tool_result)],
            "status": "success",
        }
    except Exception as e:
        return {"toolUseId": tool_use["toolUseId"], "content": [{"text": str(e)}], "status": "error"}


def tool_result_content_block(result: Any) -> ToolResultContentBlockTypeDef:
    match result:
        case str():
            return {"text": result}
        case int() | float():
            return {"text": str(result)}
        case list():
            return {"json": {"items": [str(r) for r in result]}}
        case dict():
            return {"json": result}
        case BaseModel():
            return {"json": result.model_dump()}
        case _:
            return {"text": str(result)}
