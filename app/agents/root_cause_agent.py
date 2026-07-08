"""RootCauseAnalysisAgent — links an anomalous/target metric to likely drivers.

This is a heuristic (not causal-inference) agent: given a target numeric
column, it ranks the other numeric columns by absolute correlation and
frames them as "candidate drivers" worth investigating, plus flags whether
extreme values in the target coincide with rows flagged by the
AnomalyDetectionAgent.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.agents.base_agent import BaseAgent


class RootCauseAgent(BaseAgent):
    name = "root_cause_analysis_agent"
    description = "Ranks candidate drivers behind changes in a target metric using correlation and anomaly overlap."

    def _execute(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        correlation: dict[str, Any] | None = None,
        anomaly_result: dict[str, Any] | None = None,
        top_n: int = 5,
    ) -> dict[str, Any]:
        candidate_drivers = []
        if correlation:
            for pair in correlation.get("top_pairs", []):
                if target_column in (pair["column_a"], pair["column_b"]):
                    other = pair["column_b"] if pair["column_a"] == target_column else pair["column_a"]
                    candidate_drivers.append({
                        "driver": other,
                        "correlation": pair["correlation"],
                        "strength": pair["strength"],
                    })
        candidate_drivers.sort(key=lambda d: abs(d["correlation"]), reverse=True)
        candidate_drivers = candidate_drivers[:top_n]

        anomaly_overlap = None
        if anomaly_result and anomaly_result.get("top_anomalies"):
            values_in_anomalies = [
                row["values"].get(target_column)
                for row in anomaly_result["top_anomalies"]
                if target_column in row.get("values", {})
            ]
            values_in_anomalies = [v for v in values_in_anomalies if v is not None]
            if values_in_anomalies:
                anomaly_overlap = {
                    "n_anomalous_rows_with_target": len(values_in_anomalies),
                    "avg_target_value_in_anomalies": round(sum(values_in_anomalies) / len(values_in_anomalies), 4),
                    "overall_avg_target_value": round(float(dataframe[target_column].mean()), 4),
                }

        narrative = self._narrative(target_column, candidate_drivers, anomaly_overlap)
        return {
            "target_column": target_column,
            "candidate_drivers": candidate_drivers,
            "anomaly_overlap": anomaly_overlap,
            "narrative": narrative,
        }

    @staticmethod
    def _narrative(target: str, drivers: list[dict[str, Any]], anomaly_overlap: dict[str, Any] | None) -> str:
        if not drivers:
            return f"No strongly correlated columns were found for '{target}' — root cause is likely outside this dataset."
        top = drivers[0]
        parts = [
            f"'{top['driver']}' shows the strongest relationship with '{target}' "
            f"({top['strength']}, r={top['correlation']})."
        ]
        if anomaly_overlap:
            parts.append(
                f"Anomalous rows average {anomaly_overlap['avg_target_value_in_anomalies']} for '{target}' vs. "
                f"an overall average of {anomaly_overlap['overall_avg_target_value']}, suggesting anomalies "
                "materially shift this metric."
            )
        return " ".join(parts)
