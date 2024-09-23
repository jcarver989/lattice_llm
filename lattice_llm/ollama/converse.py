from typing import Optional, Type, TypeVar

from mypy_boto3_bedrock_runtime.type_defs import ConverseOutputTypeDef
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message
from ollama import Message as OllamaMessage
from ollama import Options, chat
from pydantic import BaseModel
from .models import ModelId


def converse(
    model_id: ModelId, prompt: str, messages: list[Message], options: Optional[Options] = None
) -> ConverseOutputTypeDef:
    response = chat(model=model_id.value, messages=_format_messages(messages, prompt), options=options)

    return {"message": {"role": response["message"]["role"], "content": [{"text": response["message"]["content"]}]}}


T = TypeVar("T", bound=BaseModel, covariant=True)


def converse_with_structured_output(
    model_id: ModelId, messages: list[Message], output_schema: Type[T], options: Optional[Options] = None
) -> T:

    prompt: OllamaMessage = {
        "role": "user",
        "content": f"""Use the previous messages to populdate the JSON schema defined below. 

        # BEGIN JSON SCHEMA
        {output_schema.model_json_schema()}
        # END JSON SCHEMA
        """,
    }

    response = chat(
        model=model_id.value, messages=_format_messages(messages) + [prompt], format="json", options=options
    )

    obj = output_schema.model_validate_json(response["message"]["content"])
    return obj


def _format_messages(messages: list[Message], prompt: Optional[str] = None) -> list[OllamaMessage]:
    formatted_messages: list[OllamaMessage] = [{"role": "system", "content": prompt}] if prompt else []

    for message in messages:
        for content_block in message["content"]:
            formatted_messages.append({"role": message["role"], "content": content_block["text"]})

    return formatted_messages
