from __future__ import annotations

from paper_alpha_agent.llm.client import LLMClient
from paper_alpha_agent.models.idea import PrototypeSpec, ResearchIdea


def build_prototypes(ideas: list[ResearchIdea], llm_client: LLMClient) -> list[PrototypeSpec]:
    return [llm_client.build_prototype_spec(idea).prototype for idea in ideas]
