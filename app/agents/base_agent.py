"""Base class for every InsightForge-AI agent.

Design goals (matching the project's agent-based architecture requirement):
  * Clear responsibility — one agent, one job, implemented in `_execute()`.
  * Clean interface — every agent exposes the same `.run(**kwargs)` method.
  * Independent execution — agents don't reach into each other; they take
    plain inputs (DataFrames, dicts) and return plain outputs (dicts).
  * Logging — every run is timed and logged, success or failure.
  * Error handling — exceptions are caught, logged, and returned as a
    structured error result instead of crashing the whole pipeline.
"""
from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.utils.logger import get_logger


@dataclass
class AgentResult:
    """Uniform return type for every agent run."""

    agent_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
        }


class BaseAgent(ABC):
    """Common scaffolding shared by all concrete agents."""

    name: str = "base_agent"
    description: str = "Base agent"

    def __init__(self):
        self.logger = get_logger(f"agent.{self.name}")

    @abstractmethod
    def _execute(self, **kwargs) -> dict[str, Any]:
        """Agent-specific logic. Must return a JSON-serializable dict."""
        raise NotImplementedError

    def run(self, **kwargs) -> AgentResult:
        """Execute the agent with timing, logging, and error containment."""
        start = time.perf_counter()
        self.logger.info(f"Starting '{self.name}'")
        try:
            data = self._execute(**kwargs)
            duration = (time.perf_counter() - start) * 1000
            self.logger.info(f"Completed '{self.name}' in {duration:.1f}ms")
            return AgentResult(agent_name=self.name, success=True, data=data, duration_ms=duration)
        except Exception as exc:  # noqa: BLE001 — agents must never crash the pipeline
            duration = (time.perf_counter() - start) * 1000
            self.logger.error(f"Agent '{self.name}' failed: {exc}\n{traceback.format_exc()}")
            return AgentResult(
                agent_name=self.name, success=False, error=str(exc), duration_ms=duration
            )
