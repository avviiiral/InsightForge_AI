"""AnomalyDetectionAgent — multivariate anomaly scoring via Isolation Forest.

Complements `OutlierDetectionAgent` (which looks at one column at a time)
by finding rows that are unusual across *combinations* of numeric columns.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.agents.base_agent import BaseAgent


class AnomalyDetectionAgent(BaseAgent):
    name = "anomaly_detection_agent"
    description = "Detects multivariate anomalies across numeric columns using Isolation Forest."

    def _execute(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        contamination: float = 0.05,
        max_rows_returned: int = 20,
    ) -> dict[str, Any]:
        numeric_columns = numeric_columns or list(dataframe.select_dtypes(include=np.number).columns)
        numeric_columns = [c for c in numeric_columns if dataframe[c].nunique(dropna=True) > 1]

        if len(numeric_columns) < 2 or len(dataframe) < 20:
            return {
                "anomaly_count": 0,
                "anomaly_indices": [],
                "note": "Need at least 2 numeric columns and 20+ rows for multivariate anomaly detection.",
            }

        subset = dataframe[numeric_columns].copy()
        subset = subset.fillna(subset.median(numeric_only=True))

        model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
        )
        preds = model.fit_predict(subset)
        scores = model.score_samples(subset)

        anomaly_mask = preds == -1
        anomaly_indices = subset.index[anomaly_mask].tolist()
        ranked = sorted(zip(anomaly_indices, scores[anomaly_mask]), key=lambda t: t[1])

        top_rows = []
        for idx, score in ranked[:max_rows_returned]:
            row = dataframe.loc[idx, numeric_columns].to_dict()
            row = {k: (round(float(v), 4) if pd.notna(v) else None) for k, v in row.items()}
            top_rows.append({"row_index": int(idx), "anomaly_score": round(float(score), 4), "values": row})

        return {
            "anomaly_count": int(anomaly_mask.sum()),
            "anomaly_pct": round(100 * anomaly_mask.sum() / len(subset), 2),
            "columns_analyzed": numeric_columns,
            "top_anomalies": top_rows,
        }
