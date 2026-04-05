# paper_alpha_agent

`paper_alpha_agent` is a bounded research pipeline for scanning recent arXiv papers about machine learning and deep learning applied to financial forecasting and relative value, then turning the strongest ideas into lightweight prototype specs and toy backtests.

It is intentionally not a free-running autonomous agent. The code is organized as explicit stages with typed models, clear interfaces, and dependency boundaries so the pipeline is easy to test and extend.

## What it does

The current backbone can:

- query the arXiv export API for finance/ML-related topics
- parse and deduplicate papers into typed domain models
- score papers for relevance, implementability, and novelty using a mock LLM interface
- enrich top papers with stubbed prior-art analysis
- extract research ideas and prototype specifications
- fetch sample market data through an abstract market data interface
- run a toy backtest through an abstract backtest runner
- write a Markdown report to `data/reports/`

## Current limitations

- LLM calls are mocked and return structured placeholder outputs
- Semantic Scholar integration is stubbed
- Market data uses a dummy local generator rather than a production data vendor
- Backtesting is deliberately simple and meant only to validate wiring
- Git initialization could not be completed from this environment because local `git` is blocked by an unaccepted Xcode license on the host machine

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
```

## Run

Full pipeline:

```bash
paper-alpha-agent run
```

Discovery only:

```bash
paper-alpha-agent discover
```

Write a sample report without calling arXiv:

```bash
paper-alpha-agent report
```

You can also run the module directly:

```bash
python -m paper_alpha_agent.main run
```

## Configuration

Base settings live in:

- `config/settings.yaml`
- `config/topics.yaml`
- `config/prompts.yaml`

Environment variables override YAML values using the `PAPER_ALPHA_AGENT__...` prefix. Examples are in `.env.example`.

API keys belong in `.env`, for example:

```dotenv
PAPER_ALPHA_AGENT__API_KEYS__OPENAI=...
PAPER_ALPHA_AGENT__API_KEYS__SEMANTIC_SCHOLAR=...
```

## Replacing the mock LLM layer later

The LLM boundary is concentrated in:

- `src/paper_alpha_agent/llm/client.py`
- `src/paper_alpha_agent/llm/schemas.py`
- `src/paper_alpha_agent/llm/prompts.py`

To switch to real ChatGPT-based calls later:

1. implement a real `LLMClient` subclass in `llm/client.py`
2. load prompt templates from `config/prompts.yaml`
3. call the model with `response_format` or schema-constrained outputs
4. validate model responses through the existing Pydantic schema layer
5. keep orchestration unchanged so stages remain deterministic and testable

## Git setup

This session could not run `git` because the host machine has not accepted the Xcode license yet. After fixing that locally, run:

```bash
git init
git remote add origin https://github.com/kirmatho-gpt/paper_alpha_agent.git
git branch -M main
```
