from streamlit_flow import streamlit_flow
from streamlit_flow.layouts import TreeLayout

import streamlit as st

from ..graph import Graph
from .util import map_edges, map_nodes


def render_graph(graph: Graph) -> None:
    if "nodes" or "edges" not in st.session_state:
        st.session_state["nodes"] = []
        st.session_state["edges"] = []

    nodes = map_nodes(graph, st.session_state.last_nodes_executed)
    edges = map_edges(graph, nodes[0].id)

    st.session_state["nodes"] = nodes
    st.session_state["edges"] = edges

    streamlit_flow(
        str(st.session_state.last_nodes_executed),
        nodes,
        edges,
        layout=TreeLayout(direction="down"),
        fit_view=True,
        height=1000,
        style={"width": "100%"},
    )
