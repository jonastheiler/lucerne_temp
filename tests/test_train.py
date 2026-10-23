import pandas as pd
import pytest

from lucerne_temp.monitor import check
from lucerne_temp.train import MIN_TRAINING_DAYS, folds, gate


def metrics(lightgbm: float, raw: float = 0.75, hourly_bias: float = 0.45) -> dict:
    return {
        "raw": {"pinball": raw},
        "hourly_bias": {"pinball": hourly_bias},
        "lightgbm": {"pinball": lightgbm},
    }


def test_folds_train_strictly_before_each_test_month():
    timestamps = pd.Series(pd.date_range("2024-03-01", "2026-10-13", freq="h", tz="UTC"))
    splits = folds(timestamps)

    assert len(splits) == 20  # March 2025 to October 2026, after a year of training
    for train, test in splits:
        assert timestamps[train].max() < timestamps[test].min() - pd.Timedelta(hours=23)
        assert (timestamps[train].max() - timestamps.min()).days >= MIN_TRAINING_DAYS - 2
    tested = pd.concat([timestamps[test] for _, test in splits])
    assert tested.is_unique and tested.is_monotonic_increasing


def test_gate_promotes_a_better_model():
    assert gate(metrics(0.40), {"backtest": metrics(0.41)}) == []


@pytest.mark.parametrize(
    ("new", "champion", "reason"),
    [(0.46, None, "hourly_bias"), (0.44, {"backtest": metrics(0.40)}, "current model")],
)
def test_gate_rejects(new, champion, reason):
    problems = gate(metrics(new), champion)
    assert len(problems) == 1 and reason in problems[0]


def live(mae: float, coverage: float, n: int = 600) -> dict:
    return {"mae": mae, "coverage_80": coverage, "n": n}


@pytest.mark.parametrize(
    ("model", "problems"),
    [
        (live(1.2, 0.8), 0),
        (live(1.7, 0.8), 1),  # worse than raw
        (live(1.55, 0.8), 0),  # a near-tie with raw isn't a reason to retrain
        (live(1.2, 0.5), 1),  # interval too narrow
        (live(1.2, 0.97, n=188), 0),  # too few hours to judge coverage
        (live(1.6, 0.5, n=24), 0),  # too few hours to judge
    ],
)
def test_monitor_check(model, problems):
    assert len(check(model, live(1.5, 0.0))) == problems
