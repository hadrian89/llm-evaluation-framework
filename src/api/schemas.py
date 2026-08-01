"""Pydantic request/response schemas for the evaluation API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalCaseIn(BaseModel):
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    cases: list[EvalCaseIn]


class MetricResultOut(BaseModel):
    name: str
    score: float
    passed: bool
    threshold: float | None = None
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class EvalResultOut(BaseModel):
    case_id: str
    question: str
    answer: str
    passed: bool
    metrics: list[MetricResultOut]


class EvaluateResponse(BaseModel):
    run_id: str
    num_cases: int
    pass_rate: float
    metrics: dict[str, Any]
    results: list[EvalResultOut]


class CompareRequest(BaseModel):
    label_a: str = "model_a"
    label_b: str = "model_b"
    cases_a: list[EvalCaseIn]
    cases_b: list[EvalCaseIn]


class MetricComparisonOut(BaseModel):
    metric: str
    mean_a: float
    mean_b: float
    delta: float
    winner: str


class CompareResponse(BaseModel):
    label_a: str
    label_b: str
    pass_rate_a: float
    pass_rate_b: float
    comparisons: list[MetricComparisonOut]


class ConsistencyRequest(BaseModel):
    question: str
    answers: list[str]


class ConsistencyResponse(BaseModel):
    score: float
    passed: bool
    threshold: float
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
