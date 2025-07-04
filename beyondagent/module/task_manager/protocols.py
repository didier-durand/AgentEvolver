from typing import Any, Protocol


class LlmClient(Protocol):
    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> str:
        ...