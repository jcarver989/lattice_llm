from lattice_llm.dev_server.mappers import _get_return_ids


def f(x: int):
    if x == 0:
        return "a"
    else:
        return "b"


def y(x: int):
    return "a" if x == 0 else "b"


def z(x: int):
    if x == 0:
        return f
    else:
        return y


def test_get_return_ids_literals() -> None:
    assert _get_return_ids(f) == ["a", "b"]


def test_get_return_ids_inline_literals() -> None:
    assert _get_return_ids(y) == ["a", "b"]


def test_get_return_ids_functions() -> None:
    assert _get_return_ids(z) == ["f", "y"]
