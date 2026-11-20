"""Process-local state. Immutable criteria are isolated by caller user ID."""
from threading import RLock

from .domain import SearchCriteria


class InMemoryConversationStore:
    def __init__(self):
        self._states: dict[str, SearchCriteria] = {}
        self._lock = RLock()

    def get(self, user_id: str) -> SearchCriteria | None:
        with self._lock:
            return self._states.get(user_id)

    def set(self, user_id: str, criteria: SearchCriteria) -> None:
        with self._lock:
            self._states[user_id] = criteria

    def reset(self, user_id: str) -> None:
        with self._lock:
            self._states.pop(user_id, None)
