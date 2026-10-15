import pandas as pd

from lucerne_temp.data import save
from lucerne_temp.site import build


def test_build_shows_tomorrow_and_scores(tmp_path):
    hours = pd.date_range("2026-10-13 22:00", periods=48, freq="h", tz="UTC")  # 14-15 Oct, local
    issued = pd.Timestamp("2026-10-13 09:15", tz="UTC")
    forecasts = pd.DataFrame(
        {"timestamp": hours, "temperature_2m": 12.0, "q10": 10.0, "q50": 11.0, "q90": 13.0}
    ).assign(issued_at=[issued] * 24 + [issued + pd.Timedelta(days=1)] * 24)
    forecasts.loc[forecasts.index[-1], "q50"] = 15.0
    save(forecasts, "forecasts", tmp_path / "data")
    observations = pd.DataFrame({"timestamp": hours[:24], "temperature": 11.5})
    save(observations, "observations", tmp_path / "data")

    page = build(tmp_path / "data", tmp_path / "site").read_text(encoding="utf-8")

    assert "{{" not in page
    assert "Thursday 15 October" in page  # the newest forecast's local day
    assert "<b>11.0</b> and <b>15.0 °C</b>" in page
    assert "Not judged yet: 24 of 168 hours scored." in page
    assert "<td>0.50</td><td>0.50</td><td>100%</td>" in page
