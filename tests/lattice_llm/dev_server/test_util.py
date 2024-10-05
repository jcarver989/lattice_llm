from inspect import getsource
from os import makedirs
from os.path import dirname
from pathlib import Path
from tempfile import TemporaryDirectory

from lattice_llm.dev_server.util import _get_module_name_from_path, load_graph_from_file


def load_graph():
    from dataclasses import dataclass
    from typing import Callable

    from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef as Message

    from lattice_llm.graph import END, Graph
    from lattice_llm.graph.execution import LoadedGraph
    from lattice_llm.state import LocalStateStore

    @dataclass
    class Context:
        user_id: str
        tools: list[Callable]

    @dataclass
    class State:
        messages: list[Message]

    def say_hello(context: Context, state: State) -> State:
        return State(messages=state.messages + [{"role": "assistant", "content": [{"text": "Hi"}]}])

    graph = Graph[Context, State](nodes=[say_hello], edges=[(say_hello, END)])

    context = Context(user_id="1", tools=[])
    store = LocalStateStore(lambda: State(messages=[]))

    return LoadedGraph(graph, context, store)


def assert_graph_loads(files: dict[str, str], entrypoint: str) -> None:
    root = TemporaryDirectory()

    for file, contents in files.items():
        print(dirname(f"{root.name}/{file}"))
        makedirs(dirname(f"{root.name}/{file}"), exist_ok=True)
        with open(f"{root.name}/{file}", "w") as f:
            f.write(contents)

    file = f"{root.name}/{entrypoint}"
    loaded_graph = load_graph_from_file(file)
    graph = loaded_graph.graph
    context = loaded_graph.context
    state = loaded_graph.store.get("1")

    assert loaded_graph.context.user_id == "1"
    assert state.messages == []

    results = graph.execute(context, state)
    assert results.state.messages == [{"role": "assistant", "content": [{"text": "Hi"}]}]


def test_get_module_from_path() -> None:
    root = TemporaryDirectory()
    makedirs(f"{root.name}/foo/boo")

    Path(f"{root.name}/foo/__init__.py").touch()

    Path(f"{root.name}/foo/boo/__init__.py").touch()

    Path(f"{root.name}/foo/boo/main.py").touch()

    result = _get_module_name_from_path(f"{root.name}/foo/boo/main.py")

    assert result == "foo.boo.main"


def test_load_graph_from_file() -> None:
    assert_graph_loads(
        {"foo/__init__.py": "", "foo/boo/__init__.py": "", "foo/boo/main.py": getsource(load_graph)}, "foo/boo/main.py"
    )


def test_load_graph_from_no_init_file() -> None:
    assert_graph_loads({"foo/main.py": getsource(load_graph)}, "foo/main.py")
