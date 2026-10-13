from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
REPORT_DIR = Path("reports")

# MeteoSwiss station Luzern (LUZ), 454 m a.s.l.
STATION = "luz"
LATITUDE, LONGITUDE = 47.036, 8.301
LOCAL_TZ = "Europe/Zurich"

# ECMWF's global model (~25 km grid). Its archive at Open-Meteo is complete from March 2024.
NWP_MODEL = "ecmwf_ifs025"
NWP_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
    "pressure_msl",
    "precipitation",
]

# Every day at this UTC hour we forecast each hour of the next local day.
ISSUE_HOUR_UTC = 9

QUANTILES = [0.1, 0.5, 0.9]
