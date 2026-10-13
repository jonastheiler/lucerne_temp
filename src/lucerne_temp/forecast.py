"""Daily job: update the data and forecast every hour of tomorrow (local time).

    python -m lucerne_temp.forecast

Appends to data/forecasts/ and writes forecasts/latest.json. Forecasts are published as
files rather than through an API: they change once a day, so a server would add cost
and moving parts without making them any fresher.
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from lucerne_temp.config import DATA_DIR, LOCAL_TZ
from lucerne_temp.data import fetch_latest_forecast, load, save, update
from lucerne_temp.model import COLUMNS, features, load_model, predict

LATEST = Path("forecasts/latest.json")
# Stale observations would quietly degrade the forecast, so stop instead.
MAX_OBSERVATION_AGE = pd.Timedelta(hours=6)


def tomorrow(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    local_day = df["timestamp"].dt.tz_convert(LOCAL_TZ).dt.date
    return df[local_day == (now.tz_convert(LOCAL_TZ) + pd.Timedelta(days=1)).date()]


def forecast(now: pd.Timestamp, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    observations, archive = load("observations", data_dir), load("archive", data_dir)
    age = now - observations["timestamp"].max()
    if age > MAX_OBSERVATION_AGE:
        raise RuntimeError(f"latest observation is {age} old")

    nwp = tomorrow(fetch_latest_forecast(), now)
    X = features(nwp, observations, archive)
    pred = predict(load_model(), X)
    return pd.concat([nwp[["timestamp", "temperature_2m"]], pred], axis=1).assign(issued_at=now)


def publish(df: pd.DataFrame, path: Path = LATEST) -> None:
    path.parent.mkdir(exist_ok=True)
    records = [
        {
            "time": row.timestamp.tz_convert(LOCAL_TZ).isoformat(),
            "ecmwf": round(row.temperature_2m, 1),
            **{c: round(getattr(row, c), 1) for c in COLUMNS},
        }
        for row in df.itertuples()
    ]
    payload = {"issued_at": f"{df['issued_at'].iloc[0]:%Y-%m-%dT%H:%MZ}", "hours": records}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    today = date.today()
    update(today - timedelta(days=3), today)

    df = forecast(pd.Timestamp(datetime.now(UTC)))
    save(df, "forecasts")
    publish(df)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
