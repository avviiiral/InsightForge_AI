"""ForecastAgent — lightweight, dependency-friendly time-series forecasting.

Strategy:
  1. Aggregate the target numeric column onto a regular time index.
  2. Detect a plausible seasonal period (7 for daily/weekly, 12 for
     monthly/yearly) by checking how many periods of history exist.
  3. Fit Holt-Winters Exponential Smoothing when there's enough history for
     seasonality; otherwise fall back to a simple linear trend
     extrapolation. This keeps the agent usable on small datasets instead
     of throwing errors.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.agents.base_agent import BaseAgent


class ForecastAgent(BaseAgent):
    name = "forecast_agent"
    description = "Forecasts future values of a numeric column using time-series methods."

    def _execute(
        self,
        dataframe: pd.DataFrame,
        date_column: str,
        value_column: str,
        periods_ahead: int = 6,
        freq: str | None = None,
    ) -> dict[str, Any]:
        data = dataframe[[date_column, value_column]].dropna().copy()
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column]).sort_values(date_column)
        if len(data) < 5:
            return {"error": "Need at least 5 valid date/value rows to forecast."}

        inferred_freq = freq or self._infer_frequency(data[date_column])
        series = data.set_index(date_column)[value_column].resample(inferred_freq).mean()
        series = series.interpolate(limit_direction="both")

        if len(series) < 4:
            return {"error": "Not enough regular time periods to forecast after resampling."}

        seasonal_period = self._seasonal_period(inferred_freq, len(series))
        method_used, forecast_values, fitted_values = self._fit_and_forecast(
            series, periods_ahead, seasonal_period
        )

        future_index = pd.date_range(
            start=series.index[-1], periods=periods_ahead + 1, freq=inferred_freq
        )[1:]

        history = [
            {"period": str(idx.date()), "value": round(float(v), 4), "is_forecast": False}
            for idx, v in series.items()
        ]
        forecast = [
            {"period": str(idx.date()), "value": round(float(v), 4), "is_forecast": True}
            for idx, v in zip(future_index, forecast_values)
        ]

        return {
            "method": method_used,
            "frequency": inferred_freq,
            "seasonal_period": seasonal_period,
            "history": history,
            "forecast": forecast,
            "series_data": history + forecast,
        }

    # -- internals -----------------------------------------------------

    @staticmethod
    def _infer_frequency(dates: pd.Series) -> str:
        diffs = dates.sort_values().diff().dropna()
        if diffs.empty:
            return "D"
        median_days = diffs.dt.days.median()
        if median_days <= 1:
            return "D"
        if median_days <= 8:
            return "W"
        if median_days <= 32:
            return "MS"
        if median_days <= 100:
            return "QS"
        return "YS"

    @staticmethod
    def _seasonal_period(freq: str, n_points: int) -> int | None:
        candidates = {"D": 7, "W": 52, "MS": 12, "QS": 4, "YS": None}
        period = candidates.get(freq)
        if period and n_points >= period * 2:
            return period
        return None

    def _fit_and_forecast(
        self, series: pd.Series, periods_ahead: int, seasonal_period: int | None
    ) -> tuple[str, np.ndarray, np.ndarray]:
        if seasonal_period and len(series) >= seasonal_period * 2 + 2:
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                model = ExponentialSmoothing(
                    series, trend="add", seasonal="add", seasonal_periods=seasonal_period,
                    initialization_method="estimated",
                ).fit()
                forecast = model.forecast(periods_ahead)
                return "holt_winters_seasonal", forecast.values, model.fittedvalues.values
            except Exception:  # noqa: BLE001
                pass

        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(series, trend="add", initialization_method="estimated").fit()
            forecast = model.forecast(periods_ahead)
            return "holt_linear_trend", forecast.values, model.fittedvalues.values
        except Exception:  # noqa: BLE001
            pass

        # Final fallback: ordinary least squares linear trend.
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        future_x = np.arange(len(series), len(series) + periods_ahead)
        forecast = np.polyval(coeffs, future_x)
        fitted = np.polyval(coeffs, x)
        return "linear_regression_fallback", forecast, fitted
