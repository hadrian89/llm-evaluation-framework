# 🔍 LLM Evaluation Framework

> A comprehensive framework for evaluating LLM outputs in production — faithfulness scoring, hallucination detection, response relevance, latency benchmarking, and cost tracking. Built for teams shipping AI agents to production.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-evaluation-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

Most teams building LLM applications have no systematic way to measure output quality before shipping. This framework provides **automated evaluation pipelines** that run on every deployment, catching regressions in response quality, faithfulness, and safety before they reach users.

Born from production experience evaluating banking AI agents, where hallucinated financial advice isn't just a bad user experience — it's a compliance risk.

## What It Evaluates

| Dimension | Metrics | Method |
|-----------|---------|--------|
| **Faithfulness** | Grounding score, source attribution accuracy | LLM-as-judge + RAGAS |
| **Relevance** | Answer relevance, context precision/recall | Embedding similarity + LLM scoring |
| **Hallucination** | Fabrication rate, unsupported claim detection | Claim decomposition + fact verification |
| **Safety** | PII leakage, toxicity, off-topic rate | Rule-based + classifier |
| **Performance** | Latency p50/p95/p99, token usage, cost per query | Instrumented tracing |
| **Consistency** | Response variance across identical inputs | Multi-run statistical analysis |

## Project Structure

```
llm-evaluation-framework/
├── src/
│   ├── evaluators/
│   │   ├── faithfulness.py         # Grounding & source attribution
│   │   ├── relevance.py            # Answer & context relevance
│   │   ├── hallucination.py        # Claim extraction & verification
│   │   ├── safety.py               # PII, toxicity, compliance checks
│   │   ├── performance.py          # Latency & cost tracking
│   │   └── consistency.py          # Multi-run variance analysis
│   ├── datasets/
│   │   ├── builder.py              # Evaluation dataset construction
│   │   ├── golden_set.py           # Golden Q&A set management
│   │   └── synthetic.py            # Synthetic test case generation
│   ├── pipeline/
│   │   ├── runner.py               # Evaluation pipeline orchestrator
│   │   ├── reporter.py             # HTML & JSON report generation
│   │   └── comparator.py           # A/B model comparison
│   ├── integrations/
│   │   ├── langsmith.py            # LangSmith trace integration
│   │   └── mlflow.py               # MLflow experiment logging
│   ├── api/
│   │   ├── main.py                 # Evaluation API endpoints
│   │   └── schemas.py
│   └── config/
│       ├── settings.py
│       └── eval_config.yaml        # Evaluation thresholds & params
├── tests/
│   ├── unit/
│   └── integration/
├── examples/
│   ├── evaluate_rag.py             # RAG system evaluation example
│   ├── evaluate_agent.py           # Agent evaluation example
│   └── compare_models.py           # Model comparison example
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── eval.yml                # Scheduled evaluation runs
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

```bash
git clone https://github.com/hadrian89/llm-evaluation-framework.git
cd llm-evaluation-framework
pip install -e ".[dev]"

# Run evaluation on a RAG system
python examples/evaluate_rag.py --config config/eval_config.yaml

# Compare two models
python examples/compare_models.py --model-a gpt-4 --model-b claude-3
```

## Tech Stack

Python · RAGAS · LangSmith · LangChain · MLflow · FastAPI · pandas · pytest

## License

MIT — see [LICENSE](LICENSE).

## Author

**Abhinav Srivastav** — [LinkedIn](https://www.linkedin.com/in/abhinav-srivastav/) · [Medium](https://medium.com/@ErAbhinavSri) · [GitHub](https://github.com/hadrian89)
