from abc import abstractmethod
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class StateStore(Protocol, Generic[T]):
    @abstractmethod
    def get(self, key: str) -> T: ...

    @abstractmethod
    def set(self, key: str, state: T) -> None: ...
