import numpy as np
import pandas as pd
import pytest

from lucerne_temp.config import NWP_VARIABLES
from lucerne_temp.model import features, fit, issue_time, pinball, predict, score, training_set


def make_history(hours: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic observations and forecasts: the weather model is 2 °C too warm, plus noise."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    forecast = 10 + 8 * np.sin(2 * np.pi * timestamps.hour / 24) + rng.normal(0, 3, hours)
    archive = pd.DataFrame({v: rng.normal(0, 1, hours) for v in NWP_VARIABLES})
    archive["timestamp"], archive["temperature_2m"] = timestamps, forecast
    observations = pd.DataFrame(
        {"timestamp": timestamps, "temperature": forecast - 2 + rng.normal(0, 1, hours)}
    )
    return observations, archive


@pytest.mark.parametrize(
    "local_target",
    ["2026-03-29 00:00", "2026-03-29 23:00", "2026-10-25 00:00", "2026-10-25 23:00"],
)
def test_issue_time_is_the_previous_day_on_daylight_saving_days(local_target):
    target = pd.Series([pd.Timestamp(local_target, tz="Europe/Zurich").tz_convert("UTC")])
    expected = pd.Timestamp(local_target[:10], tz="UTC") - pd.Timedelta(hours=15)

    assert issue_time(target).iloc[0] == expected


def test_features_only_use_observations_known_at_issue_time():
    observations, archive = make_history(24 * 10)
    local_day = archive["timestamp"].dt.tz_convert("Europe/Zurich").dt.date
    target = archive[local_day == pd.Timestamp("2025-01-09").date()]
    before = features(target, observations, archive)

    # Corrupt everything measured from issue time onwards: no feature may change.
    late = observations["timestamp"] >= issue_time(target["timestamp"]).min()
    observations.loc[late, "temperature"] = 100.0
    after = features(target, observations, archive)

    pd.testing.assert_frame_equal(before, after)


def test_pinball():
    y = np.array([10.0, 10.0])
    # Under-forecasting by 2 costs 0.9 * 2 at q=0.9; over-forecasting costs 0.1 * 2.
    assert pinball(y, np.array([8.0, 12.0]), 0.9) == pytest.approx((1.8 + 0.2) / 2)


def test_fit_learns_bias_and_calibrated_interval():
    observations, archive = make_history(24 * 120)
    model = fit(training_set(observations, archive))

    observations, archive = make_history(24 * 30, seed=1)
    test = archive.merge(observations, on="timestamp")
    pred = predict(model, features(archive, observations, archive))
    metrics = score(test["temperature"], pred)

    assert (pred["q50"] - test["temperature_2m"]).mean() == pytest.approx(-2, abs=0.2)
    assert metrics["coverage_80"] == pytest.approx(0.8, abs=0.05)
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()
