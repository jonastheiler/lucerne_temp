"""Build the public dashboard: one static page with tomorrow's forecast and the live scores.

    python -m lucerne_temp.site

Writes site/index.html. The daily workflow publishes it with GitHub Pages.
"""

import json
from pathlib import Path

import pandas as pd

from lucerne_temp.config import DATA_DIR, LOCAL_TZ
from lucerne_temp.data import load
from lucerne_temp.monitor import (
    COVERAGE_MIN_HOURS,
    MIN_HOURS,
    WINDOW,
    check,
    daily,
    recent_scores,
    scored,
)

SITE_DIR = Path("site")
TEMPLATE = Path(__file__).with_name("site.html")
CHART_DAYS = 14


def tomorrow(forecasts: pd.DataFrame) -> pd.DataFrame:
    """The newest forecast: every hour of the day it was issued for."""
    return forecasts[forecasts["issued_at"] == forecasts["issued_at"].max()]


def chart_rows(forecasts: pd.DataFrame, observations: pd.DataFrame) -> list[dict]:
    start = forecasts["timestamp"].max() - pd.Timedelta(days=CHART_DAYS)
    df = observations.merge(forecasts, on="timestamp", how="outer")
    df = df[df["timestamp"] > start].sort_values("timestamp")
    values = df[["temperature", "temperature_2m", "q10", "q50", "q90"]].round(1)
    values.columns = ["obs", "ecmwf", "q10", "q50", "q90"]
    times = df["timestamp"].dt.tz_convert(LOCAL_TZ).map(pd.Timestamp.isoformat)
    return [
        {"t": t, **{k: v for k, v in row.items() if pd.notna(v)}}
        for t, row in zip(times, values.to_dict("records"), strict=True)
    ]


def status(model: dict[str, float], raw: dict[str, float]) -> tuple[str, str]:
    """A CSS class and one sentence on how the model is doing."""
    if model["n"] < MIN_HOURS:
        return "wait", f"Not judged yet: {model['n']} of {MIN_HOURS} hours scored."
    problems = check(model, raw)
    if problems:
        return "bad", "Degraded: " + "; ".join(problems) + "."
    if model["n"] < COVERAGE_MIN_HOURS:
        return (
            "ok",
            "Healthy: better than raw ECMWF. The uncertainty range is judged after 3 weeks.",
        )
    return "ok", "Healthy: better than raw ECMWF, with a well-sized uncertainty range."


def build(data_dir: Path = DATA_DIR, site_dir: Path = SITE_DIR) -> Path:
    forecasts, observations = load("forecasts", data_dir), load("observations", data_dir)
    next_day = tomorrow(forecasts)
    day = next_day["timestamp"].dt.tz_convert(LOCAL_TZ).iloc[0]
    df = scored(data_dir)
    model, raw = recent_scores(df)
    state, sentence = status(model, raw)

    days = "\n".join(
        f"<tr><td>{d}</td><td>{r.model:.2f}</td><td>{r.raw:.2f}</td><td>{r.inside:.0%}</td></tr>"
        for d, r in daily(df).iterrows()
    )
    issued = next_day["issued_at"].iloc[0].tz_convert(LOCAL_TZ)
    fields = {
        "DAY": f"{day:%A} {day.day} {day:%B}",
        "HIGH": f"{next_day['q50'].max():.1f}",
        "LOW": f"{next_day['q50'].min():.1f}",
        "STATE": state,
        "STATUS": sentence,
        "WINDOW": str(WINDOW.days),
        "MAE": f"{model['mae']:.2f}",
        "MAE_RAW": f"{raw['mae']:.2f}",
        "COVERAGE": f"{model['coverage_80']:.0%}",
        "HOURS": str(model["n"]),
        "MIN_HOURS": str(MIN_HOURS),
        "DAYS": days,
        "ISSUED": f"{issued.day} {issued:%B %Y, %H:%M}",
        "DATA": json.dumps(chart_rows(forecasts, observations)).replace("</", "<\\/"),
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in fields.items():
        page = page.replace(f"{{{{{key}}}}}", value)

    site_dir.mkdir(exist_ok=True)
    path = site_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    print(f"wrote {build()}")


if __name__ == "__main__":
    main()
