import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from lattice_llm.graph.execution import LoadedGraph


def _get_module_name_from_path(filename: str) -> str:
    p = Path(filename).resolve()
    paths = []
    if p.name != "__init__.py":
        paths.append(p.stem)
    while True:
        p = p.parent
        if not p:
            break
        if not p.is_dir():
            break

        inits = [f for f in p.iterdir() if f.name == "__init__.py"]
        if not inits:
            break

        paths.append(p.stem)

    return ".".join(reversed(paths))


def load_graph_from_file(file: str) -> LoadedGraph:
    module_name = _get_module_name_from_path(file)
    spec = spec_from_file_location(module_name, file)
    if spec and spec.loader:
        module = module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.load_graph()
    else:
        raise FileNotFoundError()
