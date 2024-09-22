import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
from ..graph import Graph, START, END
import inspect
import ast


def render_graph(graph: Graph) -> None:
    def get_style(node_id: str) -> dict[str, str]:
        if node_id in st.session_state.last_nodes_executed:
            return {"backgroundColor": "green"}
        else:
            return {"backgroundColor": "white"}

    if "nodes" or "edges" not in st.session_state:
        st.session_state["nodes"] = []
        st.session_state["edges"] = []

    st.session_state["nodes"] = [
        StreamlitFlowNode(
            id=id,
            pos=(0, 0),
            data={"content": f"""### {id}"""},
            node_type="default",
            source_position="bottom",
            target_position="top",
            style=get_style(id),
        )
        for id, node in graph.nodes.items()
    ]

    st.session_state["nodes"].append(
        StreamlitFlowNode(
            id=START,
            pos=(0, 0),
            data={"content": f"""### {START}"""},
            node_type="input",
            source_position="bottom",
            target_position="top",
        )
    )

    st.session_state["nodes"].append(
        StreamlitFlowNode(
            id=END,
            pos=(0, 0),
            data={"content": f"""### {END}"""},
            node_type="default",
            source_position="bottom",
            target_position="top",
        ),
    )

    edges = [StreamlitFlowEdge(f"{START}-{st.session_state['nodes'][0].id}", START, st.session_state["nodes"][0].id)]
    for source_id, destinations in graph.edges.items():
        for destination in destinations:
            match destination:
                case str():
                    edges.append(StreamlitFlowEdge(f"{source_id}-{destination}", source_id, destination))
                case destination if callable(destination):
                    if graph.nodes.get(destination.__name__):
                        destination_id = destination.__name__
                        edges.append(StreamlitFlowEdge(f"{source_id}-{destination_id}", source_id, destination_id))
                    else:
                        destination_ids = get_return_ids(destination)
                        for destination_id in destination_ids:
                            edges.append(
                                StreamlitFlowEdge(
                                    f"{source_id}-{destination_id}",
                                    source_id,
                                    destination_id,
                                    animated=True,
                                    label=f"{destination.__name__}",
                                )
                            )

    streamlit_flow(
        str(st.session_state.last_nodes_executed),
        st.session_state["nodes"],
        edges,
        layout=TreeLayout(direction="down"),
        fit_view=True,
        height=1000,
        style={"width": "100%"},
    )


def get_return_ids(f):
    """Does some nasty AST hackery to figure out what nodes a conditional edge returns"""

    def get_ids(elt) -> list[str]:
        if isinstance(elt, (ast.Tuple,)):
            return [str(x.id) for x in elt.elts if isinstance(x, (ast.Name,))]
        elif isinstance(elt, (ast.Name,)):
            return [str(elt.id)]
        else:
            return []

    (tree,) = ast.parse(inspect.getsource(f)).body
    returns = set[str]()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Return,)):
            for r in get_ids(node.value):
                returns.add(r)

    return returns
