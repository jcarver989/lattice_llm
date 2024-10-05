from typing import Generator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lattice_llm.bedrock import text
from lattice_llm.graph.execution import ChatbotState, GraphExecutionResult, LoadedGraph, run_graph

from .mappers import map_edges, map_messages, map_nodes
from .models import ExecuteResult
from .util import load_graph_from_file

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_loaded_graph() -> LoadedGraph:
    return app.state.loaded_graph


def _execute_graph(user_message: Optional[str] = None) -> GraphExecutionResult[ChatbotState]:
    generator: Generator[GraphExecutionResult[ChatbotState], None, None] = app.state.graph_generator

    if user_message:
        loaded_graph = _get_loaded_graph()
        user_id = loaded_graph.context.user_id
        store = loaded_graph.store
        state = store.get(user_id)
        state.messages = state.messages + [text(user_message)]
        store.set(user_id, state)

    result = next(generator)

    return result


@app.get("/graph/load")
def load(file: str):
    loaded_graph = load_graph_from_file(file)
    app.state.loaded_graph = loaded_graph
    app.state.graph_generator = run_graph(
        loaded_graph.graph, loaded_graph.context, loaded_graph.store, loaded_graph.context.user_id
    )
    return {"message": f"graph in {file} loaded!"}


@app.get("/graph/execute")
def execute(user_message: Optional[str] = None) -> ExecuteResult:
    graph = _get_loaded_graph().graph
    result = _execute_graph(user_message)

    nodes = map_nodes(graph, result.nodes_executed)
    edges = map_edges(graph)
    messages = map_messages(result.state)

    return ExecuteResult(nodes=nodes, edges=edges, messages=messages)
