from enum import Enum

from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message


class Color(Enum):
    CYAN = "\033[96m"
    GREEN = "\033[92m"


def print_message(message: Message) -> None:
    blocks = [block["text"] for block in message["content"] if block.get("text")]
    text_blocks = "\n".join(blocks)

    match message["role"]:
        case "assistant":
            print(f"{color_text('Assistant: ', Color.CYAN)} {text_blocks}")
        case "user":
            print(f"{color_text('User: ', Color.GREEN)} {text_blocks}")
    print("\n")


def color_text(text: str, color: Color) -> str:
    return f"{color.value}{text}\033[0m"
