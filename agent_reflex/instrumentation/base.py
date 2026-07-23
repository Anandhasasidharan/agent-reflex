from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgentAdapter(ABC):
    @abstractmethod
    def instrument_agent_executor(self, executor: Any) -> Any:
        ...

    @abstractmethod
    def extract_agent_id(self, executor: Any) -> str:
        ...

    @abstractmethod
    def extract_tools(self, executor: Any) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_task_graph(self, executor: Any) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_framework_name(self) -> str:
        ...
