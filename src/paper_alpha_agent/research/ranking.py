from __future__ import annotations

import logging

from paper_alpha_agent.llm.client import LLMClient
from paper_alpha_agent.models.paper import Paper, RankedPaper


LOGGER = logging.getLogger(__name__)


FINANCE_TERMS = {
    "forecast": 2.0,
    "forecasting": 2.0,
    "predict": 2.0,
    "prediction": 2.0,
    "nowcast": 1.5,
    "signal": 1.5,
    "alpha": 1.5,
    "factor": 1.5,
    "factors": 1.5,
    "risk premium": 1.5,
    "premia": 1.5,
    "spread": 1.5,
    "mispricing": 1.5,
    "mean reversion": 2.0,
    "momentum": 2.0,
    "carry": 2.0,
    "term structure": 1.5,
    "curve": 1.0,
    "cross-section": 1.5,
    "cross-sectional": 1.5,
    "relative value": 2.5,
    "statistical arbitrage": 2.5,
    "pair trading": 2.0,
    "turnover": 1.0,
    "return": 2.0,
    "returns": 2.0,
    "excess return": 2.0,
    "price": 2.0,
    "prices": 2.0,
    "yield": 2.0,
    "yields": 2.0,
    "volatility": 2.0,
    "realized volatility": 2.0,
    "implied volatility": 2.0,
    "drawdown": 1.0,
    "sharpe ratio": 1.5,
    "asset": 1.5,
    "assets": 1.5,
    "market": 1.5,
    "trading": 1.5,
    "futures": 2.0,
    "forward": 1.5,
    "forwards": 1.5,
    "equity": 1.5,
    "equities": 1.5,
    "stock": 1.0,
    "stocks": 1.0,
    "fixed income": 1.5,
    "bond": 1.5,
    "bonds": 1.5,
    "treasury": 1.5,
    "credit": 1.5,
    "option": 1.5,
    "options": 1.5,
    "derivative": 1.5,
    "derivatives": 1.5,
    "commodities": 1.5,
    "electricity": 1.5,
    "oil": 1.0,
    "gas": 1.0,
    "metals": 1.0,
    "fx": 1.5,
    "foreign exchange": 1.5,
    "currency": 1.0,
    "currencies": 1.0,
    "crypto": 1.5,
    "cryptocurrency": 1.5,
    "bitcoin": 1.0,
    "ethereum": 1.0,
    "high-frequency": 1.5,
    "low-frequency": 1.5,
    "long-term": 1.0,
    "daily": 1.0,
    "intraday": 1.0,
    "weekly": 0.5,
    "monthly": 0.5,
    "time series": 1.5,
    "regime": 2.0,
    "regime shift": 2.5,
    "latent state": 2.0,
    "feature": 0.5,
    "features": 0.5,
    "embedding": 0.5,
    "transformer": 1.0,
    "attention": 1.0,
    "neural": 1.0,
    "deep learning": 1.0,
    "machine learning": 1.0,
    "lstm": 1.0,
    "rnn": 1.0,
    "gru": 1.0,
    "xgboost": 0.75,
    "random forest": 0.75,
}

NEGATIVE_TERMS = {
    "risk": -2.0,
    "asset management": -2.0,
    "portfolio optimization": -2.0,
    "allocation": -1.0,
    "contract design": -2.0, 
    "execution": -1.0,
    "rough volatility": -2.0,
    "rHeston": -2.0,
    "rBergomi": -2.0,
    "options pricing": -1.5,
}


def heuristic_relevance_score(paper: Paper) -> float:
    text = f"{paper.title} {paper.abstract}".lower()
    score = 0.0

    if any(category.startswith("q-fin") for category in paper.categories):
        score += 3.0
    if any(category in {"cs.LG", "cs.AI", "stat.ML"} for category in paper.categories):
        score += 1.5

    for term, weight in FINANCE_TERMS.items():
        if term in text:
            score += weight
    for term, penalty in NEGATIVE_TERMS.items():
        if term in text:
            score += penalty

    return round(score, 3)


def prefilter_papers(
    papers: list[Paper],
    min_score: float = 2.0,
    keep_at_least: int = 10,
) -> list[Paper]:
    scored = sorted(
        ((paper, heuristic_relevance_score(paper)) for paper in papers),
        key=lambda item: item[1],
        reverse=True,
    )
    for paper, score in scored:
        decision = "keep" if score >= min_score else "drop"
        LOGGER.info(
            "Prefilter score=%s decision=%s paper_id=%s title=%s",
            score,
            decision,
            paper.paper_id,
            paper.title,
        )

    kept = [paper for paper, score in scored if score >= min_score]
    if len(kept) < min(keep_at_least, len(scored)):
        LOGGER.info(
            "Prefilter fallback engaged: only %s papers met min_score=%s, keeping top %s by heuristic score",
            len(kept),
            min_score,
            min(keep_at_least, len(scored)),
        )
        kept = [paper for paper, _ in scored[: min(keep_at_least, len(scored))]]

    LOGGER.info(
        "Prefilter completed: input=%s kept=%s min_score=%s keep_at_least=%s",
        len(papers),
        len(kept),
        min_score,
        keep_at_least,
    )
    return kept


def shortlist_papers_for_ranking(
    papers: list[Paper],
    shortlist_size: int,
    min_score: float = 5.0,
) -> list[Paper]:
    prefiltered = prefilter_papers(papers, min_score=min_score, keep_at_least=shortlist_size)
    shortlisted = sorted(prefiltered, key=heuristic_relevance_score, reverse=True)[:shortlist_size]
    for rank, paper in enumerate(shortlisted, start=1):
        LOGGER.info(
            "Shortlist rank=%s score=%s paper_id=%s title=%s",
            rank,
            heuristic_relevance_score(paper),
            paper.paper_id,
            paper.title,
        )
    return shortlisted


def rank_papers(papers: list[Paper], llm_client: LLMClient, relevance_threshold: float = 0.0) -> list[RankedPaper]:
    ranked: list[RankedPaper] = []
    for paper in papers:
        response = llm_client.rank_paper_relevance(paper)
        item = RankedPaper(
            **paper.model_dump(),
            **response.model_dump(),
        )
        if item.relevance_score >= relevance_threshold:
            ranked.append(item)
    ranked.sort(key=lambda paper: paper.composite_score, reverse=True)
    return ranked
