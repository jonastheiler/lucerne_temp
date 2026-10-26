"""Download observations and archived forecasts, validate them and store them as Parquet.

observations: measured temperature, the truth for training, backtest and monitoring
archive:      ECMWF's day-ahead forecasts for past hours, the inputs for training

python -m lucerne_temp.data                         # the last 3 days
python -m lucerne_temp.data 2024-03-01 2026-10-12   # backfill
"""

import argparse
import io
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from lucerne_temp.config import (
    DATA_DIR,
    LATITUDE,
    LONGITUDE,
    NWP_MODEL,
    NWP_VARIABLES,
    STATION,
)

# Measured temperatures from MeteoSwiss's open data (ogd-smn = SwissMetNet stations).
# {station} and {period} are filled in by fetch_observations.
OBS_URL = (
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/{station}/ogd-smn_{station}_t_{period}.csv"
)
# ECMWF's newest forecast for the coming days, used to forecast tomorrow.
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Old forecasts for past hours, used for training and the backtest.
# For a past hour, "previous_day1" is the forecast made about 24 hours earlier: what we
# would have known when forecasting the next day. The latest forecast for a past hour is
# a few hours old and would make the backtest look better than reality.
ARCHIVE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_SUFFIX = "_previous_day1"

# Free public APIs have the odd hiccup. Retry only errors that may pass on their own
# (429 rate limit, 5xx server trouble), not our own mistakes like 404.
# The pauses grow (0, 4, 8, 16, 32 s), so a struggling server gets time to recover.
RETRY = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
# Every request in this file goes through this session, so both APIs get the retries.
HTTP = requests.Session()
HTTP.mount("https://", HTTPAdapter(max_retries=RETRY))

# Plausible ranges, wide on purpose: they catch broken data (a dead sensor, a unit
# change), not rare weather. A value outside them is a data error.
# The two temperatures are required: measured minus forecast is what the model learns.
# The measurement is also the truth that the backtest and live monitoring score against,
# so one bad value would mislead training and scoring at once.
# The other weather model outputs may have gaps: LightGBM handles missing inputs.
REQUIRED = ["temperature", "temperature_2m"]
BOUNDS = {
    "observations": {"temperature": (-30, 45)},
    "archive": {
        "temperature_2m": (-30, 45),
        "dew_point_2m": (-40, 35),
        "cloud_cover": (0, 100),
        "wind_speed_10m": (0, 200),
        "pressure_msl": (900, 1100),
        "precipitation": (0, 200),
    },
}


def parse_observations(text: str) -> pd.DataFrame:
    # tre200s0 is MeteoSwiss's code for air temperature 2 m above ground, instantaneous value.
    df = pd.read_csv(io.StringIO(text), sep=";", usecols=["reference_timestamp", "tre200s0"])
    # An explicit format: guessing could swap day and month (03.04.) without any error.
    df["timestamp"] = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M", utc=True)
    # The file has a value every 10 minutes. The instantaneous value at the full hour
    # matches what the weather model outputs; an hourly mean would be a different quantity.
    # Gaps (sensor maintenance) are dropped; the model can cope with a missing hour.
    df = df[df["timestamp"].dt.minute == 0].dropna(subset="tre200s0")
    # From here on, the project's own names: the rest of the code never sees MeteoSwiss's.
    return df.rename(columns={"tre200s0": "temperature"})[["timestamp", "temperature"]]


def fetch_observations(start: date, end: date) -> pd.DataFrame:
    """Hourly temperature at the station. MeteoSwiss splits it into three files."""
    # One file per decade up to last year, then "recent" (since 1 January) and "now" (24h).
    # Empty if start is in the current year: then only "recent" and "now".
    decades = sorted({year // 10 * 10 for year in range(start.year, date.today().year)})
    # Oldest file first, freshest last: keep="last" below relies on this order.
    periods = [f"historical_{d}-{d + 9}" for d in decades] + ["recent", "now"]

    frames = []
    for period in periods:
        response = HTTP.get(OBS_URL.format(station=STATION, period=period), timeout=120)
        response.raise_for_status()
        frames.append(parse_observations(response.text))
    # The files overlap at their edges; for an hour in two files, keep the fresher value.
    df = pd.concat(frames).drop_duplicates("timestamp", keep="last")
    return between(df, start, end)


def parse_forecast(payload: dict, suffix: str = "") -> pd.DataFrame:
    # One list per variable: {"hourly": {"time": [...], "temperature_2m": [...]}}.
    hourly = dict(payload["hourly"])
    # Without the suffix, archive and live tables have the same column names,
    # so the model finds the inputs it was trained on.
    df = pd.DataFrame(hourly).drop(columns="time").rename(columns=lambda c: c.removesuffix(suffix))
    # The times have no zone but are UTC (we ask for GMT): tz_localize attaches the
    # label without shifting the clock. Local times here would be off by 1-2 h, silently.
    df.insert(0, "timestamp", pd.to_datetime(hourly["time"]).tz_localize("UTC"))
    return df


def get_forecast(url: str, suffix: str, **params) -> pd.DataFrame:
    # One function for both APIs, so training and live use the same model, place and
    # variables. params: the caller's time range, e.g. {"forecast_days": 3}.
    params |= {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": NWP_MODEL,
        # The archive needs the suffix on every name: "temperature_2m_previous_day1,...".
        "hourly": ",".join(v + suffix for v in NWP_VARIABLES),
        "timezone": "GMT",  # UTC like the observations; parse_forecast relies on it
    }
    response = HTTP.get(url, params=params, timeout=120)
    response.raise_for_status()
    return parse_forecast(response.json(), suffix)


def fetch_archive(start: date, end: date) -> pd.DataFrame:
    """Forecasts as they were issued about one day ahead, for past hours only."""
    df = get_forecast(
        ARCHIVE_URL, ARCHIVE_SUFFIX, start_date=start.isoformat(), end_date=end.isoformat()
    )
    # For future hours the archive falls back to the latest forecast, so cut at "now".
    now = pd.Timestamp.now(tz="UTC")
    # Drop only rows with no forecast at all; single gaps are fine for LightGBM.
    return df[df["timestamp"] <= now].dropna(how="all", subset=NWP_VARIABLES)


def fetch_latest_forecast() -> pd.DataFrame:
    # Days count from today 00:00 UTC. 2 would cover tomorrow in Lucerne; 3 leaves room
    # for a late run after local midnight, when "tomorrow" is the day after.
    return get_forecast(FORECAST_URL, "", forecast_days=3)


def between(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Rows from `start` 00:00 UTC up to and including the whole of `end`."""
    # A date compares as its midnight, so "<= end" would lose 23 hours of the last day.
    # Half-open [lo, hi) like range(a, b): up to, not including, the next midnight.
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end + timedelta(days=1), tz="UTC")
    return df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)].reset_index(drop=True)


def validate(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> None:
    """Fail loudly instead of silently storing bad data."""
    # Collect every problem and raise once, so one failed run shows them all.
    problems = []
    if df.empty:  # an API can answer 200 OK with no data
        problems.append("no rows")
    # A guard rail for the parsing above: all later code relies on UTC.
    if str(df["timestamp"].dt.tz) != "UTC":
        problems.append("timestamps are not UTC")
    if df["timestamp"].duplicated().any():
        problems.append("duplicate timestamps")
    for column, (low, high) in bounds.items():
        # Gaps are allowed, except in the two required temperatures.
        values = df[column] if column in REQUIRED else df[column].dropna()
        if not values.between(low, high).all():  # NaN also fails
            problems.append(f"{column} missing or outside [{low}, {high}]")
    if problems:
        raise ValueError("; ".join(problems))


def save(df: pd.DataFrame, name: str, data_dir: Path = DATA_DIR) -> None:
    """Upsert into one Parquet file per month, so re-running a day never duplicates rows."""
    # Monthly files: a daily run rewrites one small file, and git never re-stores old months.
    for month, part in df.groupby(df["timestamp"].dt.strftime("%Y-%m")):
        path = data_dir / name / f"{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # New rows below old ones, so keep="last" stores the newest download.
            part = pd.concat([pd.read_parquet(path), part])
            part = part.drop_duplicates("timestamp", keep="last")
        part.sort_values("timestamp").to_parquet(path, index=False)


def load(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # Reads all monthly files in the folder as one table.
    return pd.read_parquet(data_dir / name).sort_values("timestamp").reset_index(drop=True)


def update(start: date, end: date, data_dir: Path = DATA_DIR) -> None:
    frames = {"observations": fetch_observations(start, end), "archive": fetch_archive(start, end)}

    # Validate everything before writing anything: a bad archive must not leave
    # observations half-updated on disk.
    for name, df in frames.items():
        validate(df, BOUNDS[name])
    for name, df in frames.items():
        save(df, name, data_dir)
        print(f"{name}: {len(df)} rows up to {df['timestamp'].max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", nargs="?", type=date.fromisoformat)
    parser.add_argument("end", nargs="?", type=date.fromisoformat)
    args = parser.parse_args()

    # No dates = daily mode: re-fetch the last 3 days so late corrections and
    # missed runs heal themselves (save() keeps the freshest copy).
    today = date.today()
    update(args.start or today - timedelta(days=3), args.end or today)


if __name__ == "__main__":
    main()
