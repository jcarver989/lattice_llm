from typing import Sequence
from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef, MessageOutputTypeDef
from lattice_llm.bedrock.client import FakeBedrockModel, FakeBedrockClient, fake_converse_response
from lattice_llm.bedrock import ModelId
from lattice_llm.bedrock import converse


class FakeClaud(FakeBedrockModel):
    id = ModelId.CLAUDE_3_5

    def generate_response(self, messages: Sequence[MessageUnionTypeDef]) -> MessageOutputTypeDef:
        return {"role": "assistant", "content": [{"text": "Hello"}]}


def test_convese() -> None:
    client = FakeBedrockClient([FakeClaud()])
    response = converse(client, ModelId.CLAUDE_3_5, "You're an LLM", [])
    assert response == fake_converse_response({"role": "assistant", "content": [{"text": "Hello"}]})
