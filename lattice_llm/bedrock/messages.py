from typing import Any

from mypy_boto3_bedrock_runtime.literals import ConversationRoleType
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message


def text(text_blocks: str | list[str], role: ConversationRoleType = "user") -> Message:
    match text_blocks:
        case list():
            return {"role": role, "content": [{"text": text} for text in text_blocks]}
        case str():
            return {"role": role, "content": [{"text": text_blocks}]}


def tool_result(id: str, results: dict[str, Any]) -> Message:
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": id, "content": [{"text": str(results)}], "status": "success"}}],
    }
