"""Phase 1 Step 2: fetch conditions from Open-Meteo (free, no key).

fetch_weather: one hour, live.  fetch_weather_history: a date range in the past (Phase 3).
"""
import pandas as pd
import requests

FORECAST = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"  # the live one only reaches ~3 months back
SATELLITE = "https://satellite-api.open-meteo.com/v1/archive"
AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"

FORECAST_VARS = ("temperature_2m", "relative_humidity_2m", "surface_pressure", "cloud_cover",
                 "total_column_integrated_water_vapour", "direct_radiation", "diffuse_radiation")
SATELLITE_VARS = ("shortwave_radiation", "direct_radiation", "diffuse_radiation")


def _get(url, lat, lon, variables, **when):
    """Hourly `variables` as a DataFrame indexed by UTC hour. `when`: start_hour/end_hour or start_date/end_date."""
    r = requests.get(url, timeout=60, params={
        "latitude": lat, "longitude": lon, "hourly": ",".join(variables), "timezone": "UTC", **when})
    r.raise_for_status()
    hourly = r.json()["hourly"]
    return pd.DataFrame({v: hourly[v] for v in variables},
                        index=pd.to_datetime(hourly["time"]).tz_localize("UTC"), dtype=float)


def _combine(forecast, satellite, air):
    """Forecast conditions + measured satellite sunlight (forecast as fallback) + haze."""
    w = forecast.copy()
    measured = satellite["direct_radiation"].notna()
    for v in ("direct_radiation", "diffuse_radiation"):
        w[v] = satellite[v].where(measured, w[v])
    w["radiation_source"] = measured.map({True: "satellite", False: "forecast"})
    w["aod_550nm"] = air["aerosol_optical_depth"]
    return w


def fetch_weather(lat, lon, hour):
    """Conditions for `hour` (UTC) as a dict. Radiation values are means over the hour *ending* at `hour`.

    Units: °C, %, hPa, %, kg/m² (= mm water), W/m², unitless AOD at 550 nm.
    """
    h = hour.strftime("%Y-%m-%dT%H:%M")
    when = {"start_hour": h, "end_hour": h}
    forecast = _get(FORECAST, lat, lon, FORECAST_VARS, **when)
    try:
        satellite = _get(SATELLITE, lat, lon, SATELLITE_VARS, **when)
    except requests.HTTPError as e:
        if e.response.status_code != 400:  # 400 = hour not covered yet (satellite data ends today)
            raise
        satellite = pd.DataFrame(index=forecast.index, columns=list(SATELLITE_VARS), dtype=float)
    w = _combine(forecast, satellite, _get(AIR_QUALITY, lat, lon, ("aerosol_optical_depth",), **when))
    return {k: (None if pd.isna(v) else v) for k, v in w.iloc[0].items()}


def fetch_weather_history(lat, lon, start_date, end_date):
    """Hourly conditions between two dates ("YYYY-MM-DD", inclusive) as a DataFrame."""
    when = {"start_date": start_date, "end_date": end_date}
    return _combine(_get(HISTORICAL_FORECAST, lat, lon, FORECAST_VARS, **when),
                    _get(SATELLITE, lat, lon, SATELLITE_VARS, **when),
                    _get(AIR_QUALITY, lat, lon, ("aerosol_optical_depth",), **when))
