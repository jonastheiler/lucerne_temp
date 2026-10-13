import pandas as pd
import pytest

from lucerne_temp.data import BOUNDS, parse_forecast, parse_observations, save, validate

CSV = """station_abbr;reference_timestamp;tre200s0;ure200s0
LUZ;13.10.2026 09:50;15.9;64
LUZ;13.10.2026 10:00;16.1;63
LUZ;13.10.2026 11:00;;61
LUZ;13.10.2026 12:00;17.0;60
"""


def make_archive(**overrides) -> pd.DataFrame:
    row = {
        "temperature_2m": 12.0,
        "dew_point_2m": 8.0,
        "cloud_cover": 50.0,
        "wind_speed_10m": 5.0,
        "pressure_msl": 1015.0,
        "precipitation": 0.0,
    }
    timestamps = pd.date_range("2026-10-01", periods=2, freq="h", tz="UTC")
    return pd.DataFrame([row | overrides] * 2).assign(timestamp=timestamps)


def test_parse_observations_keeps_full_hours_in_utc():
    df = parse_observations(CSV)

    assert list(df.columns) == ["timestamp", "temperature"]
    assert str(df["timestamp"].dt.tz) == "UTC"
    # 09:50 is not a full hour and 11:00 has no temperature.
    assert df["timestamp"].dt.hour.tolist() == [10, 12]


def test_parse_forecast_strips_suffix():
    payload = {"hourly": {"time": ["2026-10-01T00:00"], "temperature_2m_previous_day1": [12.0]}}
    df = parse_forecast(payload, "_previous_day1")

    assert list(df.columns) == ["timestamp", "temperature_2m"]
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_validate_accepts_gaps_in_optional_inputs():
    validate(make_archive(dew_point_2m=float("nan")), BOUNDS["archive"])


@pytest.mark.parametrize(
    ("column", "value"), [("temperature_2m", float("nan")), ("cloud_cover", 140.0)]
)
def test_validate_rejects_bad_values(column, value):
    with pytest.raises(ValueError, match=column):
        validate(make_archive(**{column: value}), BOUNDS["archive"])


def test_validate_rejects_duplicates():
    df = make_archive().assign(timestamp=pd.Timestamp("2026-10-01", tz="UTC"))
    with pytest.raises(ValueError, match="duplicate"):
        validate(df, BOUNDS["archive"])


def test_save_is_idempotent(tmp_path):
    df = make_archive()
    save(df, "archive", tmp_path)
    save(df, "archive", tmp_path)

    assert len(pd.read_parquet(tmp_path / "archive")) == len(df)
