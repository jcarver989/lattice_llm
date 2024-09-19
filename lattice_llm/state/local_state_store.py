from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeVar

from .state_store import StateStore

T = TypeVar("T")


class LocalStateStore(Generic[T]):
    state: dict[str, T]
    default_state: Callable[[], T]

    def __init__(
        self,
        default_state: Callable[[], T],
        initial_state: Optional[dict[str, T]] = None,
    ):
        self.state = initial_state or {}
        self.default_state = default_state

    def get(self, key: str) -> T:
        return self.state.get(key) or self.default_state()

    def set(self, key: str, state: T) -> None:
        self.state[key] = state


if TYPE_CHECKING:
    _store: StateStore[list[str]] = LocalStateStore(lambda: [])
