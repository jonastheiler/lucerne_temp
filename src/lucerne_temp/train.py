"""Backtest a new model against baselines and the current one, and promote it if it's better.

    python -m lucerne_temp.train

Writes reports/backtest.md. Only if the new model passes the gate are model/ and
model/card.json replaced; otherwise the command fails, so a scheduled run gets noticed.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from lucerne_temp.config import MODEL_DIR, QUANTILES, REPORT_DIR
from lucerne_temp.data import load
from lucerne_temp.model import COLUMNS, fit, predict, save_model, score, training_set

# Start testing once there is about a year of training data, so every season has been seen.
MIN_TRAINING_DAYS = 365
# A new model may be at most this much worse than the current one. Each retrain adds a
# month to the backtest, so the two scores are close but not identical.
TOLERANCE = 1.02


def raw(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Baseline 1: the weather model as is, with no uncertainty."""
    return pd.DataFrame({c: test["temperature_2m"] for c in COLUMNS})


def hourly_bias(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Baseline 2: add the error quantiles seen at that hour of day. A classic, simple
    correction that a model with more inputs has to beat to be worth its complexity."""
    error = train["temperature"] - train["temperature_2m"]
    table = error.groupby(train["hour"]).quantile(QUANTILES).unstack()
    table.columns = COLUMNS
    return table.loc[test["hour"]].set_index(test.index).add(test["temperature_2m"], axis=0)


def lightgbm(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return predict(fit(train), test)


METHODS = {"raw": raw, "hourly_bias": hourly_bias, "lightgbm": lightgbm}


def folds(timestamps: pd.Series) -> list[tuple[pd.Series, pd.Series]]:
    """Rolling-origin splits: train on everything before a month, test on that month.

    Training stops a day before the test month: a forecast issued the day before the
    month starts couldn't have known the observations of that last day.
    """
    first = timestamps.min() + pd.Timedelta(days=MIN_TRAINING_DAYS)
    months = pd.date_range(first.ceil("D"), timestamps.max(), freq="MS")
    return [
        (
            timestamps < month - pd.Timedelta(days=1),
            (timestamps >= month) & (timestamps < month + pd.offsets.MonthBegin()),
        )
        for month in months
    ]


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample predictions of every method, one row per method and hour."""
    results = []
    for train_mask, test_mask in folds(df["timestamp"]):
        train, test = df[train_mask], df[test_mask]
        for name, method in METHODS.items():
            pred = method(train, test).assign(method=name, timestamp=test["timestamp"])
            results.append(pred.assign(y=test["temperature"]))
    return pd.concat(results, ignore_index=True)


def summarize(results: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {name: score(g["y"], g[COLUMNS]) for name, g in results.groupby("method", sort=False)}


def gate(metrics: dict[str, dict[str, float]], champion: dict | None) -> list[str]:
    """Reasons not to promote the new model. Empty means promote."""
    new = metrics["lightgbm"]["pinball"]
    problems = [
        f"pinball {new:.3f} not better than baseline {name} ({m['pinball']:.3f})"
        for name, m in metrics.items()
        if name != "lightgbm" and new >= m["pinball"]
    ]
    if champion and new > champion["backtest"]["lightgbm"]["pinball"] * TOLERANCE:
        problems.append(
            f"pinball {new:.3f} worse than current model "
            f"({champion['backtest']['lightgbm']['pinball']:.3f})"
        )
    return problems


def report(metrics: dict[str, dict[str, float]], results: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {name} | {m['mae']:.2f} | {m['pinball']:.3f} | {m['coverage_80']:.0%} |"
        for name, m in metrics.items()
    )
    by_month = results.assign(
        month=results["timestamp"].dt.strftime("%Y-%m"), error=(results["y"] - results["q50"]).abs()
    ).pivot_table(index="month", columns="method", values="error")[list(METHODS)]
    months = "\n".join(
        f"| {month} | " + " | ".join(f"{v:.2f}" for v in row) + " |"
        for month, row in by_month.iterrows()
    )
    return f"""# Backtest

Rolling-origin backtest: each month is forecast by a model trained only on earlier data.
{results["timestamp"].min():%Y-%m-%d} to {results["timestamp"].max():%Y-%m-%d}, \
{metrics["raw"]["n"]} hours.

| Method | MAE (°C) | Pinball loss | 10-90% interval coverage (target 80%) |
|---|---|---|---|
{rows}

## MAE by month (°C)

| Month | {" | ".join(METHODS)} |
|---|{"---|" * len(METHODS)}
{months}
"""


def train(model_dir: Path = MODEL_DIR, report_dir: Path = REPORT_DIR) -> list[str]:
    df = training_set(load("observations"), load("archive"))
    results = backtest(df)
    metrics = summarize(results)

    report_dir.mkdir(exist_ok=True)
    (report_dir / "backtest.md").write_text(report(metrics, results), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    card_path = model_dir / "card.json"
    champion = json.loads(card_path.read_text()) if card_path.exists() else None
    problems = gate(metrics, champion)
    if problems:
        return problems

    # The backtest estimated how well this recipe works; the model we ship uses all data.
    model = fit(df)
    save_model(model, model_dir)
    card = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "rows": len(df),
        "interval_margin": model.margin,
        "backtest": metrics,
    }
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print("promoted new model")
    return []


def main() -> None:
    problems = train()
    if problems:
        sys.exit("not promoted: " + "; ".join(problems))


if __name__ == "__main__":
    main()
