import streamlit as st
from typing import cast
from ..graph import Graph, START
from ..graph.execution import V, W
from .render_graph import render_graph
from inspect import getsource


def run_graph_on_streamlit(graph: Graph[V, W], context: V, initial_state: W) -> None:
    st.set_page_config(layout="wide")
    st.title("Graph")

    if "graph_state" not in st.session_state:
        st.session_state["graph_state"] = initial_state

    if "graph_nodes" not in st.session_state:
        st.session_state.graph_nodes = graph.nodes

    col1, col2 = st.columns(2, gap="large")
    with col1:
        if "last_nodes_executed" not in st.session_state:
            st.session_state.last_nodes_executed = [START]

            result = graph.execute(
                context, cast(W, st.session_state.graph_state), from_node=st.session_state.last_nodes_executed
            )

            st.session_state.graph_state = result.state
            st.session_state.last_nodes_executed = result.nodes_executed

        with st.container(height=800):
            for message in st.session_state.graph_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"][0]["text"])

            st.container(height=50, border=False)  # spacer
            if prompt := st.chat_input("What is up?"):
                st.chat_message("user").markdown(prompt)
                st.session_state.graph_state.messages.append({"role": "user", "content": [{"text": prompt}]})

                with st.status("Thinking..."):
                    result = graph.execute(
                        context, cast(W, st.session_state.graph_state), from_node=st.session_state.last_nodes_executed
                    )

                    st.session_state.graph_state = result.state
                    st.session_state.last_nodes_executed = result.nodes_executed
                    st.write("Done!")

                with st.chat_message("assistant"):
                    st.markdown(result.state.messages[-1]["content"][0]["text"])

        for node_id in st.session_state.last_nodes_executed:
            st.markdown(f"### Current node's source code:")
            node = st.session_state.graph_nodes[node_id]
            st.code(getsource(node), language="python")

    with col2:
        render_graph(graph)
