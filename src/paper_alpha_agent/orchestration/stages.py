from __future__ import annotations

from enum import StrEnum


class PipelineStage(StrEnum):
    DISCOVER = "discover"
    SUMMARIZE_TOPICS = "summarize_topics"
    RANK = "rank"
    PRIOR_ART = "prior_art"
    SELECT = "select"
    EXTRACT_IDEAS = "extract_ideas"
    BUILD_PROTOTYPES = "build_prototypes"
    BACKTEST = "backtest"
    REPORT = "report"
