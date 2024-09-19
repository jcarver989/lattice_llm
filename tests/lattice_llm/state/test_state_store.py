from lattice_llm.state import LocalStateStore


def test_local_store_get_empty_state() -> None:
    store = LocalStateStore[list[str]](lambda: [])
    assert store.get("foo") == []


def test_local_store_get_populated_state() -> None:
    def state():
        return []

    store = LocalStateStore[list[str]](state)
    store.set("user-1", ["hello", "world"])

    assert store.get("user-1") == ["hello", "world"]
    assert store.get("user-2") == []
