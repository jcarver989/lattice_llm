from typing import Any, Any, Mapping, Protocol, Sequence, cast, TYPE_CHECKING
from abc import ABC

from mypy_boto3_bedrock_runtime.type_defs import (
    ConverseResponseTypeDef,
    GuardrailConfigurationTypeDef,
    InferenceConfigurationTypeDef,
    MessageOutputTypeDef,
    MessageUnionTypeDef,
    SystemContentBlockTypeDef,
    ToolConfigurationTypeDef,
)
from typing_extensions import runtime_checkable

from .models import ModelId
from abc import abstractmethod


@runtime_checkable
class BedrockClient(Protocol):
    """Interface for an AWS Bedrock Client. This allows "fake" objects to be passed in for unit tests."""

    @abstractmethod
    def converse(
        self,
        *,
        modelId: str,
        messages: Sequence[MessageUnionTypeDef],
        system: Sequence[SystemContentBlockTypeDef] = ...,
        inferenceConfig: InferenceConfigurationTypeDef = ...,
        toolConfig: ToolConfigurationTypeDef = ...,
        guardrailConfig: GuardrailConfigurationTypeDef = ...,
        additionalModelRequestFields: Mapping[str, Any] = ...,
        additionalModelResponseFieldPaths: Sequence[str] = ...,
    ) -> ConverseResponseTypeDef: ...


class FakeBedrockModel(ABC):
    id: ModelId

    @abstractmethod
    def generate_response(self, messages: Sequence[MessageUnionTypeDef]) -> MessageOutputTypeDef: ...


def empty_list() -> list[Any]:
    return []


def empty_dict() -> dict[Any, Any]:
    return {}


def empty_inference_conf() -> InferenceConfigurationTypeDef:
    return {}


def empty_tool_conf() -> ToolConfigurationTypeDef:
    return {"tools": []}


def empty_guardrails() -> GuardrailConfigurationTypeDef:
    return {
        "guardrailIdentifier": "",
        "guardrailVersion": "",
    }


class FakeBedrockClient:
    """A stub BedrockClient for testing"""

    models: dict[ModelId, FakeBedrockModel]

    def __init__(self, models: Sequence[FakeBedrockModel]):
        self.models = {}
        for model in models:
            self.models[model.id] = model

    def converse(
        self,
        *,
        modelId: str,
        messages: Sequence[MessageUnionTypeDef],
        system: Sequence[SystemContentBlockTypeDef] = empty_list(),
        inferenceConfig: InferenceConfigurationTypeDef = empty_inference_conf(),
        toolConfig: ToolConfigurationTypeDef = empty_tool_conf(),
        guardrailConfig: GuardrailConfigurationTypeDef = empty_guardrails(),
        additionalModelRequestFields: Mapping[str, Any] = empty_dict(),
        additionalModelResponseFieldPaths: Sequence[str] = empty_list(),
    ) -> ConverseResponseTypeDef:
        model = self.models[cast(ModelId, modelId)]
        message = model.generate_response(messages)
        return fake_converse_response(message)


def fake_converse_response(message: MessageOutputTypeDef) -> ConverseResponseTypeDef:
    """Convenience method for testing"""
    return {
        "output": {"message": message},
        "stopReason": "end_turn",
        "metrics": {"latencyMs": 0},
        "additionalModelResponseFields": {},
        "trace": {},
        "ResponseMetadata": {
            "RequestId": "",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {},
            "RetryAttempts": 0,
        },
        "usage": {
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
        },
    }


# Hack to force mypy to check that Fakes implement the expected interface (Protocol)
if TYPE_CHECKING:
    _client: BedrockClient = FakeBedrockClient([])
