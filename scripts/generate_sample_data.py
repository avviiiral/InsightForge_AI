"""Generates the sample datasets shipped in `data/samples/`.

Run with: python scripts/generate_sample_data.py
Re-running overwrites the existing sample files deterministically (fixed
random seed) so they're safe to regenerate at any time.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from app.config import SAMPLES_DIR

RNG = np.random.default_rng(42)


def generate_sales_sample(n_rows: int = 1500) -> pd.DataFrame:
    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    categories = ["Electronics", "Apparel", "Home & Garden", "Sports", "Toys", "Groceries"]
    channels = ["Online", "Retail Store", "Wholesale", "Marketplace"]

    dates = pd.date_range("2023-01-01", periods=730, freq="D")
    order_dates = RNG.choice(dates, size=n_rows)

    base_price = RNG.gamma(shape=2.0, scale=40, size=n_rows) + 5
    quantity = RNG.integers(1, 15, size=n_rows)
    discount_pct = RNG.choice([0, 0, 0, 5, 10, 15, 20], size=n_rows)
    revenue = base_price * quantity * (1 - discount_pct / 100)

    # Inject a mild upward trend + seasonality for forecasting demos.
    day_index = (pd.Series(order_dates) - pd.Timestamp("2023-01-01")).dt.days
    trend = 1 + day_index / 3000
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * day_index / 365)
    revenue = revenue * trend.values * seasonality.values

    df = pd.DataFrame({
        "order_id": [f"ORD-{100000 + i}" for i in range(n_rows)],
        "order_date": order_dates,
        "region": RNG.choice(regions, size=n_rows),
        "product_category": RNG.choice(categories, size=n_rows),
        "sales_channel": RNG.choice(channels, size=n_rows),
        "unit_price": np.round(base_price, 2),
        "quantity": quantity,
        "discount_pct": discount_pct,
        "revenue": np.round(revenue, 2),
        "customer_email": [f"customer{i}@example.com" for i in range(n_rows)],
        "customer_satisfaction": np.clip(RNG.normal(4.2, 0.8, size=n_rows), 1, 5).round(1),
    })

    # Inject a few realistic data-quality issues.
    missing_idx = RNG.choice(n_rows, size=int(n_rows * 0.04), replace=False)
    df.loc[missing_idx, "customer_satisfaction"] = np.nan
    dup_idx = RNG.choice(n_rows, size=10, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    # A handful of extreme outliers.
    outlier_idx = RNG.choice(df.index, size=6, replace=False)
    df.loc[outlier_idx, "revenue"] = df.loc[outlier_idx, "revenue"] * RNG.uniform(8, 15)

    return df.sort_values("order_date").reset_index(drop=True)


def generate_employee_sample(n_rows: int = 600) -> pd.DataFrame:
    departments = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Customer Support"]
    levels = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"]
    locations = ["New York", "London", "Bangalore", "Singapore", "Toronto", "Remote"]

    level_salary_base = {"Junior": 55000, "Mid": 75000, "Senior": 100000, "Lead": 125000, "Manager": 140000, "Director": 175000}
    level_choices = RNG.choice(levels, size=n_rows, p=[0.28, 0.27, 0.2, 0.12, 0.09, 0.04])
    salary_noise = RNG.normal(0, 8000, size=n_rows)
    salary = np.array([level_salary_base[lvl] for lvl in level_choices]) + salary_noise

    hire_dates = pd.date_range("2015-01-01", "2026-01-01", periods=1000)
    hire_date_choice = RNG.choice(hire_dates, size=n_rows)

    df = pd.DataFrame({
        "employee_id": [f"EMP-{5000 + i}" for i in range(n_rows)],
        "full_name_hash": [f"employee_{i}" for i in range(n_rows)],
        "department": RNG.choice(departments, size=n_rows),
        "job_level": level_choices,
        "location": RNG.choice(locations, size=n_rows),
        "hire_date": hire_date_choice,
        "annual_salary": np.round(salary, -2),
        "performance_score": np.clip(RNG.normal(3.6, 0.6, size=n_rows), 1, 5).round(2),
        "is_remote": RNG.choice([True, False], size=n_rows, p=[0.35, 0.65]),
        "work_email": [f"employee{i}@insightforge-demo.com" for i in range(n_rows)],
        "phone_number": [f"+1-555-{RNG.integers(100,999)}-{RNG.integers(1000,9999)}" for _ in range(n_rows)],
    })

    missing_idx = RNG.choice(n_rows, size=int(n_rows * 0.03), replace=False)
    df.loc[missing_idx, "performance_score"] = np.nan

    return df.sort_values("hire_date").reset_index(drop=True)


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    sales_df = generate_sales_sample()
    sales_path = SAMPLES_DIR / "sales_sample.csv"
    sales_df.to_csv(sales_path, index=False)
    print(f"Wrote {len(sales_df):,} rows to {sales_path}")

    employees_df = generate_employee_sample()
    employees_path = SAMPLES_DIR / "employees_sample.csv"
    employees_df.to_csv(employees_path, index=False)
    print(f"Wrote {len(employees_df):,} rows to {employees_path}")


if __name__ == "__main__":
    main()
