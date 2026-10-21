"""Score the live forecasts against what was measured, and fail if the model degrades.

    python -m lucerne_temp.monitor

Writes reports/monitoring.md. A failure triggers a retrain in the daily workflow.
"""

import sys
from pathlib import Path

import pandas as pd

from lucerne_temp.config import DATA_DIR, REPORT_DIR
from lucerne_temp.data import load
from lucerne_temp.model import COLUMNS, score

WINDOW = pd.Timedelta(days=28)
MIN_HOURS = 7 * 24  # don't judge the model on a handful of days
COVERAGE_RANGE = (0.65, 0.92)  # the 10-90% interval should hold about 80% of hours
COVERAGE_MIN_HOURS = 21 * 24  # a calm or a stormy week alone pushes coverage out of range


def scored(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    if not (data_dir / "forecasts").exists():
        return pd.DataFrame()
    forecasts = load("forecasts", data_dir)
    return forecasts.merge(load("observations", data_dir), on="timestamp")


def check(model: dict[str, float], raw: dict[str, float]) -> list[str]:
    """Reasons to retrain. Empty means healthy."""
    if model["n"] < MIN_HOURS:
        return []
    problems = []
    if model["mae"] >= raw["mae"]:
        problems.append(f"MAE {model['mae']:.2f} no better than raw ECMWF ({raw['mae']:.2f})")
    low, high = COVERAGE_RANGE
    if model["n"] >= COVERAGE_MIN_HOURS and not low <= model["coverage_80"] <= high:
        coverage = model["coverage_80"]
        problems.append(f"interval coverage {coverage:.0%} outside [{low:.0%}, {high:.0%}]")
    return problems


def recent_scores(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Scores of the corrected and the raw forecast over the last WINDOW."""
    recent = df[df["timestamp"] > df["timestamp"].max() - WINDOW]
    model = score(recent["temperature"], recent[COLUMNS])
    raw = score(recent["temperature"], pd.DataFrame({c: recent["temperature_2m"] for c in COLUMNS}))
    return model, raw


def daily(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    """Mean absolute errors and interval coverage per UTC day, newest first."""
    return (
        df.assign(
            day=df["timestamp"].dt.strftime("%Y-%m-%d"),
            model=(df["temperature"] - df["q50"]).abs(),
            raw=(df["temperature"] - df["temperature_2m"]).abs(),
            inside=df["temperature"].between(df["q10"], df["q90"]),
        )
        .groupby("day")[["model", "raw", "inside"]]
        .mean()
        .tail(days)
        .iloc[::-1]
    )


def monitor(data_dir: Path = DATA_DIR, report_dir: Path = REPORT_DIR) -> list[str]:
    df = scored(data_dir)
    if df.empty:
        print("no forecasts to score yet")
        return []
    model, raw = recent_scores(df)
    problems = check(model, raw)

    days = "\n".join(
        f"| {day} | {r.model:.2f} | {r.raw:.2f} | {r.inside:.0%} |"
        for day, r in daily(df).iterrows()
    )
    if model["n"] < MIN_HOURS:
        status = f"not judged yet ({model['n']} of {MIN_HOURS} hours needed)"
        print(status)
        status = f"- ⏳ {status}"
    else:
        print(*problems or ["healthy"], sep="\n")
        status = "\n".join(f"- ⚠️ {p}" for p in problems) or "- ✅ healthy"
        if model["n"] < COVERAGE_MIN_HOURS:
            pending = f"coverage not judged yet ({model['n']} of {COVERAGE_MIN_HOURS} hours needed)"
            print(pending)
            status += f"\n- ⏳ {pending}"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "monitoring.md").write_text(
        f"""# Live monitoring

Last {WINDOW.days} days, {model["n"]} scored hours.

{status}

| | MAE (°C) | Pinball loss | 10-90% interval coverage |
|---|---|---|---|
| Corrected forecast | {model["mae"]:.2f} | {model["pinball"]:.3f} | {model["coverage_80"]:.0%} |
| Raw ECMWF | {raw["mae"]:.2f} | {raw["pinball"]:.3f} | n/a |

## Last 14 days

| Day | MAE corrected | MAE raw | Coverage |
|---|---|---|---|
{days}
""",
        encoding="utf-8",
    )
    return problems


def main() -> None:
    if monitor():
        sys.exit(1)


if __name__ == "__main__":
    main()
