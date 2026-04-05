from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
ENV_PREFIX = "PAPER_ALPHA_AGENT__"


class LLMSettings(BaseModel):
    model_name: str = "mock-research-model"
    temperature: float = 0.0


class PipelineSettings(BaseModel):
    date_window_days: int = 30
    max_papers: int = 20
    top_k: int = 5
    relevance_threshold: float = 0.55


class ReportingSettings(BaseModel):
    output_dir: str = "data/reports"


class APIKeys(BaseModel):
    openai: str | None = None
    semantic_scholar: str | None = None


class AppSettings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)
    api_keys: APIKeys = Field(default_factory=APIKeys)
    arxiv_query_topics: list[str] = Field(default_factory=list)
    prompts: dict[str, str] = Field(default_factory=dict)

    @property
    def report_output_path(self) -> Path:
        return ROOT_DIR / self.reporting.output_dir


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_env_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _deep_set(target: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = target
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _load_env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        nested_keys = key.removeprefix(ENV_PREFIX).lower().split("__")
        _deep_set(overrides, nested_keys, _parse_env_value(value))
    return overrides


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    load_dotenv(ROOT_DIR / ".env", override=False)
    base = _load_yaml(CONFIG_DIR / "settings.yaml")
    env = _load_env_overrides()
    merged = _deep_merge(base, env)
    merged["arxiv_query_topics"] = _load_yaml(CONFIG_DIR / "topics.yaml").get("topics", [])
    merged["prompts"] = _load_yaml(CONFIG_DIR / "prompts.yaml")
    return AppSettings.model_validate(merged)
