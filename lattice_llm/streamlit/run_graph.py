from inspect import getsource, getsourcefile, getsourcelines
from typing import cast

from streamlit.delta_generator import DeltaGenerator

import streamlit as st

from ..graph import START, Graph, Node
from ..graph.execution import V, W
from .render_graph import render_graph


def run_graph_on_streamlit(graph: Graph[V, W], context: V, initial_state: W) -> None:
    st.set_page_config(layout="wide")
    _initialize_state(graph, context, initial_state)
    col1, col2 = st.columns(2, gap="large")

    col1.header("Chat")
    messages = col1.container(height=500)

    for message in st.session_state.graph_state.messages:
        messages.chat_message(message["role"]).markdown(message["content"][0]["text"])

    if prompt := col1.chat_input("What is up?"):
        st.session_state.graph_state.messages.append({"role": "user", "content": [{"text": prompt}]})
        messages.chat_message("user").markdown(prompt)
        status = col1.status("Thinking...")
        result = graph.execute(
            context, cast(W, st.session_state.graph_state), from_node=st.session_state.last_nodes_executed
        )

        st.session_state.graph_state = result.state
        st.session_state.last_nodes_executed = result.nodes_executed
        status.write("Done!")
        messages.chat_message("assistant").markdown(result.state.messages[-1]["content"][0]["text"])

    for node_id in st.session_state.last_nodes_executed:
        node = st.session_state.graph_nodes[node_id]
        _render_source_code(node, col1)

    col2.header("Flow Chart")
    with col2:
        render_graph(graph)


def _render_source_code(node: Node, container: DeltaGenerator) -> None:
    container.header("Current node's source code:")
    container.code(getsource(node), language="python")
    container.text(getsourcefile(node))
    container.text(getsourcelines(node))


def _initialize_state(graph: Graph[V, W], context: V, initial_state: W) -> None:
    if "graph_state" not in st.session_state:
        st.session_state["graph_state"] = initial_state

    if "graph_nodes" not in st.session_state:
        st.session_state.graph_nodes = graph.nodes

    if "last_nodes_executed" not in st.session_state:
        st.session_state.last_nodes_executed = [START]

        result = graph.execute(
            context, cast(W, st.session_state.graph_state), from_node=st.session_state.last_nodes_executed
        )

        st.session_state.graph_state = result.state
        st.session_state.last_nodes_executed = result.nodes_executed
