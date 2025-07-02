import sys
from importlib import invalidate_caches, import_module
from pathlib import Path

from lattice_llm.graph.execution import LoadedGraph


def _get_module_path(filename: str) -> tuple[str, str]:
    p = Path(filename).resolve()
    paths: list[str] = []

    if p.name != "__init__.py":
        paths.append(str(p.stem))

    p = p.parent
    while True:
        if not p or not p.is_dir():
            break

        inits = [f for f in p.iterdir() if f.name == "__init__.py"]
        if not inits:
            break

        paths.append(p.stem)
        p = p.parent

    return (str(p), ".".join(reversed(paths)))


def load_graph_from_file(file: str) -> LoadedGraph:
    parent, module_name = _get_module_path(file)
    sys.path.append(parent)
    invalidate_caches()
    module = import_module(module_name)
    return module.load_graph()
