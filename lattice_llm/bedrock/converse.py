from typing import Callable, Optional, Type, TypeVar

from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef as ConverseResponse
from mypy_boto3_bedrock_runtime.type_defs import InferenceConfigurationTypeDef as InferenceConfig
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message
from pydantic import BaseModel

from .client import BedrockClient
from .models import ModelId
from .tools import get_tool_defs


def converse(
    client: BedrockClient,
    model_id: ModelId,
    prompt: str,
    messages: list[Message],
    config: InferenceConfig = {},
    tools: Optional[list[Callable]] = None,
) -> ConverseResponse:
    if tools:
        return client.converse(
            modelId=model_id.value,
            messages=messages,
            system=[{"text": prompt}],
            inferenceConfig=config,
            toolConfig={"tools": get_tool_defs(tools)},
        )

    return client.converse(
        modelId=model_id.value,
        messages=messages,
        system=[{"text": prompt}],
        inferenceConfig=config or {},
    )


T = TypeVar("T", bound=BaseModel, covariant=True)


def converse_with_structured_output(
    client: BedrockClient,
    model_id: ModelId,
    prompt: str,
    messages: list[Message],
    output_schema: Type[T],
    config: Optional[InferenceConfig] = None,
) -> T:
    tool_name = "json_schema"
    response = client.converse(
        modelId=model_id.value,
        messages=messages,
        system=[{"text": prompt}],
        inferenceConfig=config or {},
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": f"{tool_name}",
                        "description": "Represents the JSON schema for the desired output format.",
                        "inputSchema": {"json": output_schema.model_json_schema()},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": f"{tool_name}"}},
        },
    )

    json = response["output"]["message"]["content"][0]["toolUse"]["input"]
    return output_schema.model_validate(json)
