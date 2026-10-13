"""Features, model and metrics, shared by the backtest, training and the daily forecast.

The model does not predict the temperature itself but the error of the weather model
(observed minus forecast), as quantiles. The corrected forecast is the weather model's
forecast plus that predicted error.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from lucerne_temp.config import ISSUE_HOUR_UTC, LOCAL_TZ, MODEL_DIR, NWP_VARIABLES, QUANTILES

FEATURES = [*NWP_VARIABLES, "hour", "doy_sin", "doy_cos", "recent_error"]
COLUMNS = [f"q{round(q * 100)}" for q in QUANTILES]  # q10, q50, q90

# Small trees and few rounds: stronger regularisation gave better-calibrated quantiles.
PARAMS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_data_in_leaf": 200,
    "verbose": -1,
}
ROUNDS = 200
CALIBRATION_BLOCKS = 4
COVERAGE = QUANTILES[-1] - QUANTILES[0]  # 0.8


def issue_time(target: pd.Series) -> pd.Series:
    """When the forecast for `target` is made: ISSUE_HOUR_UTC on the previous local day."""
    local_midnight = target.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    return local_midnight.dt.tz_localize("UTC") - pd.Timedelta(hours=24 - ISSUE_HOUR_UTC)


def features(nwp: pd.DataFrame, observations: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """One row per target hour, using only what is known at its issue time.

    nwp:          the weather model's forecast for the target hours
    observations: measured temperatures, for the recent error
    archive:      past day-ahead forecasts, for the recent error
    """
    df = nwp[["timestamp", *NWP_VARIABLES]].copy()
    ts = df["timestamp"]
    # UTC hour rather than local: the sun doesn't move with summer time.
    df["hour"] = ts.dt.hour
    df["doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)

    # How wrong the weather model was over the last 24 hours. Errors persist in situations
    # it handles badly, such as fog or cold air trapped under warmer air.
    error = (
        observations.set_index("timestamp")["temperature"]
        - archive.set_index("timestamp")["temperature_2m"]
    ).dropna()
    recent = error.rolling("24h", min_periods=12).mean()
    # The hour before issue time: the observation *at* issue time may not be published yet.
    known_at = issue_time(ts) - pd.Timedelta(hours=1)
    df["recent_error"] = recent.reindex(known_at).to_numpy()
    return df


def training_set(observations: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Features plus the observed temperature, for every hour with an archived forecast."""
    df = features(archive, observations, archive)
    return df.merge(observations, on="timestamp").reset_index(drop=True)


@dataclass
class Model:
    boosters: dict[float, lgb.Booster]
    margin: float  # widens the outer quantiles so the interval holds COVERAGE of the hours


def fit_boosters(df: pd.DataFrame) -> dict[float, lgb.Booster]:
    residual = df["temperature"] - df["temperature_2m"]
    dataset = lgb.Dataset(df[FEATURES], residual, free_raw_data=False)
    return {q: lgb.train(PARAMS | {"alpha": q}, dataset, ROUNDS) for q in QUANTILES}


def predict_boosters(boosters: dict[float, lgb.Booster], df: pd.DataFrame) -> np.ndarray:
    residuals = np.column_stack([boosters[q].predict(df[FEATURES]) for q in QUANTILES])
    # Separately trained quantiles can cross; sorting fixes that without changing the median.
    return df[["temperature_2m"]].to_numpy() + np.sort(residuals, axis=1)


def fit(df: pd.DataFrame) -> Model:
    """Fit the quantile models, then calibrate their interval.

    Quantile models are overconfident on data they haven't seen: their 10-90% interval
    holds fewer than 80% of new observations. Conformalized quantile regression fixes
    that: predict each time block with models trained on the other blocks, measure how
    far outside the interval the observations fall, and widen it by that margin.
    """
    y, scores = df["temperature"].to_numpy(), []
    for block in np.array_split(np.arange(len(df)), CALIBRATION_BLOCKS):
        held_out = np.zeros(len(df), dtype=bool)
        held_out[block] = True
        pred = predict_boosters(fit_boosters(df[~held_out]), df[held_out])
        scores.append(np.maximum(pred[:, 0] - y[held_out], y[held_out] - pred[:, -1]))
    margin = float(np.quantile(np.concatenate(scores), COVERAGE))
    return Model(fit_boosters(df), margin)


def predict(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    values = predict_boosters(model.boosters, df)
    values[:, 0] -= model.margin
    values[:, -1] += model.margin
    return pd.DataFrame(values, columns=COLUMNS, index=df.index)


def save_model(model: Model, model_dir: Path = MODEL_DIR) -> None:
    """Plain-text files, so a model change shows up as a readable git diff."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for q, column in zip(QUANTILES, COLUMNS, strict=True):
        model.boosters[q].save_model(model_dir / f"{column}.txt")
    (model_dir / "margin.json").write_text(json.dumps({"margin": model.margin}))


def load_model(model_dir: Path = MODEL_DIR) -> Model:
    boosters = {
        q: lgb.Booster(model_file=model_dir / f"{column}.txt")
        for q, column in zip(QUANTILES, COLUMNS, strict=True)
    }
    margin = json.loads((model_dir / "margin.json").read_text())["margin"]
    return Model(boosters, margin)


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Quantile loss: penalises being on the wrong side of the quantile, weighted by q."""
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def score(y: pd.Series, pred: pd.DataFrame) -> dict[str, float]:
    """MAE of the median, mean pinball loss, and how often y falls in the 10-90% interval."""
    y, values = y.to_numpy(), pred[COLUMNS].to_numpy()
    low, median, high = values.T
    return {
        "mae": float(np.mean(np.abs(y - median))),
        "pinball": float(np.mean([pinball(y, values[:, i], q) for i, q in enumerate(QUANTILES)])),
        "coverage_80": float(np.mean((y >= low) & (y <= high))),
        "n": len(y),
    }
