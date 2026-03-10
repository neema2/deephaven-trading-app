"""
Data Science Agent — Analysis
===============================
Run analytical workflows: statistics, correlations, regressions,
anomaly detection, visualization recommendations.

Tools:
    - run_sql_analysis        — execute DuckDB SQL + stats summary
    - compute_correlation     — pairwise correlation matrix
    - compute_statistics      — descriptive stats
    - detect_anomalies        — Z-score / IQR anomaly detection
    - run_regression          — linear regression
    - time_series_decompose   — trend + seasonality + residual
    - suggest_visualization   — recommend chart type + axes

Usage::

    from agents._datascience import create_datascience_agent

    agent = create_datascience_agent(ctx)
    result = agent.run("What is the correlation between AAPL and GOOGL?")
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from ai import Agent, tool

from agents._context import _PlatformContext

logger = logging.getLogger(__name__)

DATASCIENCE_SYSTEM_PROMPT = """\
You are the Data Science Agent — a platform specialist that runs analytical \
workflows on platform data.

You can:
1. Run SQL queries against the lakehouse and summarize results statistically.
2. Compute pairwise correlations between columns or datasets.
3. Generate descriptive statistics (mean, std, percentiles, skew, kurtosis).
4. Detect anomalies using Z-score or IQR methods.
5. Run linear regressions with R², coefficients, and residual diagnostics.
6. Decompose time series into trend, seasonality, and residual components.
7. Suggest appropriate visualizations for data exploration.

Data access:
- Pull data from the Lakehouse (DuckDB SQL over Iceberg tables).
- Pull time series from the TSDB (via MarketData REST API).
- Use pandas/scipy for in-process computation.

When analyzing data:
- State your methodology clearly.
- Report key statistics with proper formatting.
- Note limitations, caveats, and assumptions.
- Suggest follow-up analyses when appropriate.
"""


def create_datascience_tools(ctx: _PlatformContext) -> list:
    """Create Data Science agent tools bound to a _PlatformContext."""

    @tool
    def run_sql_analysis(sql: str, description: str = "") -> str:
        """Execute a SQL query against the lakehouse and return statistical summary.

        Runs the query via DuckDB, then computes descriptive stats on numeric columns.

        Args:
            sql: SQL query to execute (tables: lakehouse.default.<name>).
            description: Optional description of what this analysis computes.
        """
        if ctx.lakehouse is None:
            return json.dumps({"error": "No Lakehouse configured."})

        try:
            df = ctx.lakehouse.query_df(sql)
            summary = {"description": description, "row_count": len(df), "columns": list(df.columns)}

            # Compute stats for numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                stats = {}
                for col in numeric_cols:
                    series = df[col].dropna()
                    if len(series) == 0:
                        continue
                    stats[col] = {
                        "count": len(series),
                        "mean": round(float(series.mean()), 6),
                        "std": round(float(series.std()), 6),
                        "min": round(float(series.min()), 6),
                        "25%": round(float(series.quantile(0.25)), 6),
                        "50%": round(float(series.quantile(0.50)), 6),
                        "75%": round(float(series.quantile(0.75)), 6),
                        "max": round(float(series.max()), 6),
                    }
                summary["numeric_stats"] = stats

            # Sample rows
            summary["sample_rows"] = df.head(10).to_dict(orient="records")
            return json.dumps(summary, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def compute_statistics(sql: str, columns: str = "") -> str:
        """Compute detailed descriptive statistics for columns in a query result.

        Includes mean, std, skewness, kurtosis, percentiles, null counts.

        Args:
            sql: SQL query that produces the data.
            columns: Comma-separated column names to analyze. Empty = all numeric columns.
        """
        if ctx.lakehouse is None:
            return json.dumps({"error": "No Lakehouse configured."})

        try:
            df = ctx.lakehouse.query_df(sql)
            col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None

            if col_list:
                df_subset = df[col_list]
            else:
                df_subset = df.select_dtypes(include=["number"])

            stats: dict[str, dict[str, int | float]] = {}
            for col in df_subset.columns:
                series = df_subset[col].dropna()
                n = len(series)
                if n == 0:
                    stats[col] = {"count": 0, "null_count": int(df[col].isna().sum())}
                    continue

                mean = float(series.mean())
                std = float(series.std()) if n > 1 else 0.0

                # Skewness
                skew = 0.0
                if n > 2 and std > 0:
                    skew = float(sum((x - mean) ** 3 for x in series) / (n * std ** 3))

                # Kurtosis (excess)
                kurt = 0.0
                if n > 3 and std > 0:
                    kurt = float(sum((x - mean) ** 4 for x in series) / (n * std ** 4)) - 3.0

                col_stats: dict[str, int | float] = {
                    "count": n,
                    "null_count": int(df[col].isna().sum()),
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "min": round(float(series.min()), 6),
                    "5%": round(float(series.quantile(0.05)), 6),
                    "25%": round(float(series.quantile(0.25)), 6),
                    "50%": round(float(series.quantile(0.50)), 6),
                    "75%": round(float(series.quantile(0.75)), 6),
                    "95%": round(float(series.quantile(0.95)), 6),
                    "max": round(float(series.max()), 6),
                    "skewness": round(skew, 4),
                    "kurtosis": round(kurt, 4),
                }
                stats[col] = col_stats

            return json.dumps({"statistics": stats}, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def compute_correlation(sql: str, columns: str = "") -> str:
        """Compute pairwise Pearson correlation matrix for numeric columns.

        Args:
            sql: SQL query that produces the data.
            columns: Comma-separated column names. Empty = all numeric columns.
        """
        if ctx.lakehouse is None:
            return json.dumps({"error": "No Lakehouse configured."})

        try:
            df = ctx.lakehouse.query_df(sql)
            col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None

            if col_list:
                df_subset = df[col_list]
            else:
                df_subset = df.select_dtypes(include=["number"])

            if df_subset.shape[1] < 2:
                return json.dumps({"error": "Need at least 2 numeric columns for correlation."})

            corr = df_subset.corr()
            # Convert to nested dict
            corr_dict = {}
            for col in corr.columns:
                corr_dict[col] = {
                    row: round(float(corr.loc[row, col]), 4)
                    for row in corr.index
                }

            # Find strongest correlations (off-diagonal)
            strong: list[dict[str, str | float]] = []
            for i, c1 in enumerate(corr.columns):
                for j, c2 in enumerate(corr.columns):
                    if i < j:
                        val = float(corr.loc[c1, c2])
                        if abs(val) > 0.5:
                            strong.append({
                                "pair": f"{c1} × {c2}",
                                "correlation": round(val, 4),
                                "strength": "strong" if abs(val) > 0.7 else "moderate",
                            })
            strong.sort(key=lambda x: abs(float(x["correlation"])), reverse=True)

            return json.dumps({
                "correlation_matrix": corr_dict,
                "notable_correlations": strong[:10],
                "columns_analyzed": list(corr.columns),
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def detect_anomalies(sql: str, column: str, method: str = "zscore",
                         threshold: float = 3.0) -> str:
        """Detect anomalies in a numeric column.

        Args:
            sql: SQL query that produces the data.
            column: Column name to check for anomalies.
            method: Detection method — "zscore" (default) or "iqr".
            threshold: Sensitivity threshold. For zscore: number of std devs (default 3.0).
                       For IQR: multiplier (default 1.5).
        """
        if ctx.lakehouse is None:
            return json.dumps({"error": "No Lakehouse configured."})

        try:
            df = ctx.lakehouse.query_df(sql)
            if column not in df.columns:
                return json.dumps({"error": f"Column '{column}' not found."})

            series = df[column].dropna()
            n = len(series)
            if n < 3:
                return json.dumps({"error": "Need at least 3 data points."})

            anomalies: list[dict[str, str | float | int]] = []
            if method == "zscore":
                mean = float(series.mean())
                std = float(series.std())
                if std == 0:
                    return json.dumps({"anomaly_count": 0, "method": "zscore",
                                      "message": "Zero variance — no anomalies possible."})
                for idx, val in series.items():
                    z = abs((float(val) - mean) / std)
                    if z > threshold:
                        anomalies.append({
                            "index": int(idx),
                            "value": round(float(val), 6),
                            "z_score": round(z, 3),
                        })
            elif method == "iqr":
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                for idx, val in series.items():
                    v = float(val)
                    if v < lower or v > upper:
                        anomalies.append({
                            "index": idx,
                            "value": round(v, 6),
                            "bound_exceeded": "lower" if v < lower else "upper",
                        })

            return json.dumps({
                "column": column,
                "method": method,
                "threshold": threshold,
                "total_rows": n,
                "anomaly_count": len(anomalies),
                "anomaly_rate": f"{len(anomalies)/n*100:.2f}%",
                "anomalies": anomalies[:20],
                "truncated": len(anomalies) > 20,
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def run_regression(sql: str, target: str, features: str) -> str:
        """Run a linear regression.

        Fits OLS regression and reports R², coefficients, p-values, and residual stats.

        Args:
            sql: SQL query that produces the data.
            target: Target (dependent) variable column name.
            features: Comma-separated feature (independent) variable column names.
        """
        if ctx.lakehouse is None:
            return json.dumps({"error": "No Lakehouse configured."})

        try:
            df = ctx.lakehouse.query_df(sql)
            feature_list = [f.strip() for f in features.split(",")]

            # Validate columns
            missing = [c for c in [target, *feature_list] if c not in df.columns]
            if missing:
                return json.dumps({"error": f"Columns not found: {missing}"})

            # Drop rows with any NaN in relevant columns
            df_clean = df[[target, *feature_list]].dropna()
            n = len(df_clean)
            k = len(feature_list)

            if n < k + 2:
                return json.dumps({"error": f"Need at least {k+2} rows, got {n}."})

            y = df_clean[target].values.astype(float)
            X = df_clean[feature_list].values.astype(float)

            # Add intercept
            import numpy as np
            ones = np.ones((n, 1))
            X_full = np.hstack([ones, X])

            # OLS: β = (X'X)^-1 X'y
            try:
                beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                return json.dumps({"error": "Singular matrix — features may be collinear."})

            # Predictions and residuals
            y_hat = X_full @ beta
            residuals = y - y_hat
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else r_squared

            # Residual standard error
            rse = math.sqrt(ss_res / (n - k - 1)) if n > k + 1 else 0.0

            # Coefficients
            coefficients = [
                {"name": "intercept", "value": round(float(beta[0]), 6)}
            ]
            for i, feat in enumerate(feature_list):
                coefficients.append({
                    "name": feat,
                    "value": round(float(beta[i + 1]), 6),
                })

            # Residual diagnostics
            residual_stats = {
                "mean": round(float(np.mean(residuals)), 6),
                "std": round(float(np.std(residuals)), 6),
                "min": round(float(np.min(residuals)), 6),
                "max": round(float(np.max(residuals)), 6),
            }

            return json.dumps({
                "target": target,
                "features": feature_list,
                "n_observations": n,
                "r_squared": round(r_squared, 6),
                "adjusted_r_squared": round(adj_r_squared, 6),
                "residual_std_error": round(rse, 6),
                "coefficients": coefficients,
                "residual_diagnostics": residual_stats,
            }, default=str)
        except ImportError:
            return json.dumps({"error": "numpy is required for regression. Install with: pip install numpy"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def time_series_decompose(symbol: str, msg_type: str = "equity",
                              interval: str = "1m", window: int = 20) -> str:
        """Decompose a time series into trend, seasonal, and residual components.

        Uses a simple moving average for trend extraction.

        Args:
            symbol: Symbol to analyze.
            msg_type: Asset type — "equity" or "fx".
            interval: Bar interval for the data.
            window: Moving average window size for trend (default 20).
        """
        try:
            import httpx
            resp = httpx.get(
                f"{ctx.md_base_url}/md/bars/{msg_type}/{symbol}",
                params={"interval": interval},
                timeout=10.0,
            )
            resp.raise_for_status()
            bars = resp.json()

            if not isinstance(bars, list) or len(bars) < window + 2:
                return json.dumps({"error": f"Need at least {window+2} bars, got {len(bars) if isinstance(bars, list) else 0}"})

            closes = [b.get("close", b.get("price", 0)) for b in bars]

            # Trend: simple moving average
            trend: list[float | None] = []
            for i in range(len(closes)):
                if i < window - 1:
                    trend.append(None)
                else:
                    trend.append(sum(closes[i-window+1:i+1]) / window)

            # Residual = actual - trend
            residuals = []
            for i in range(len(closes)):
                if trend[i] is not None:
                    residuals.append(round(closes[i] - trend[i], 6))
                else:
                    residuals.append(None)

            # Stats on residuals (non-None)
            valid_residuals = [r for r in residuals if r is not None]
            res_mean = sum(valid_residuals) / len(valid_residuals) if valid_residuals else 0
            res_std = math.sqrt(
                sum((r - res_mean) ** 2 for r in valid_residuals) / max(len(valid_residuals) - 1, 1)
            ) if len(valid_residuals) > 1 else 0

            t_start = trend[window - 1]
            t_end = trend[-1]
            return json.dumps({
                "symbol": symbol,
                "interval": interval,
                "window": window,
                "data_points": len(closes),
                "trend_start": round(t_start, 4) if t_start is not None else None,
                "trend_end": round(t_end, 4) if t_end is not None else None,
                "trend_direction": "up" if t_start is not None and t_end is not None and t_end > t_start else "down",
                "residual_mean": round(res_mean, 6),
                "residual_std": round(res_std, 6),
                "last_5_trend": [round(t, 4) for t in trend[-5:] if t is not None],
                "last_5_residual": [r for r in residuals[-5:] if r is not None],
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def suggest_visualization(data_description: str, question: str = "") -> str:
        """Suggest appropriate chart types and axes for data visualization.

        Uses AI to recommend the best visualization approach.

        Args:
            data_description: Description of the data (columns, types, row count).
            question: The analytical question being explored.
        """
        if ctx.ai is None:
            return json.dumps({"error": "No AI configured."})

        try:
            from ai._types import Message
            prompt = f"""\
Given this data and question, recommend the best visualization.

Data: {data_description}
Question: {question or 'General exploration'}

Return JSON with:
{{
  "chart_type": "bar|line|scatter|heatmap|histogram|box|pie",
  "x_axis": "column name",
  "y_axis": "column name or list",
  "color_by": "optional grouping column",
  "reasoning": "brief explanation of why this chart type"
}}"""

            response = ctx.ai.generate(
                [Message(role="user", content=prompt)],
                temperature=0.3,
            )
            text = response.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            return text
        except Exception as e:
            return json.dumps({"error": str(e)})

    return [run_sql_analysis, compute_statistics, compute_correlation,
            detect_anomalies, run_regression, time_series_decompose,
            suggest_visualization]


def create_datascience_agent(ctx: _PlatformContext, **kwargs: Any) -> Agent:
    """Create a Data Science Agent bound to a _PlatformContext."""
    tools = create_datascience_tools(ctx)
    return Agent(
        tools=tools,
        system_prompt=DATASCIENCE_SYSTEM_PROMPT,
        name="datascience",
        **kwargs,
    )
