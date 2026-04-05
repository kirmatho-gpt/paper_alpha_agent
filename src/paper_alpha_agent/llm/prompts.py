from __future__ import annotations

from paper_alpha_agent.config import AppSettings


class PromptLibrary:
    def __init__(self, settings: AppSettings) -> None:
        self._prompts = settings.prompts

    def get(self, name: str) -> str:
        return self._prompts.get(name, "")
