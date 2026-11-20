"""Phase 1 Step 2: fetch one hour of live conditions from Open-Meteo (free, no key)."""
import requests

FORECAST = "https://api.open-meteo.com/v1/forecast"
SATELLITE = "https://satellite-api.open-meteo.com/v1/archive"
AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"

FORECAST_VARS = ("temperature_2m", "relative_humidity_2m", "surface_pressure", "cloud_cover",
                 "total_column_integrated_water_vapour", "direct_radiation", "diffuse_radiation")
SATELLITE_VARS = ("shortwave_radiation", "direct_radiation", "diffuse_radiation")


def _hour(url, lat, lon, hour, variables):
    """Values of `variables` for one UTC hour (pandas Timestamp on the hour)."""
    h = hour.strftime("%Y-%m-%dT%H:%M")
    r = requests.get(url, timeout=20, params={
        "latitude": lat, "longitude": lon, "hourly": ",".join(variables),
        "timezone": "UTC", "start_hour": h, "end_hour": h,
    })
    r.raise_for_status()
    hourly = r.json()["hourly"]
    return {v: hourly[v][0] for v in variables}


def fetch_weather(lat, lon, hour):
    """Conditions for `hour` (UTC). Radiation values are means over the hour *ending* at `hour`.

    Units: °C, %, hPa, %, kg/m² (= mm water), W/m², unitless AOD at 550 nm.
    """
    w = _hour(FORECAST, lat, lon, hour, FORECAST_VARS)
    sat = _hour(SATELLITE, lat, lon, hour, SATELLITE_VARS)
    # Satellite readings are the real measurement; fall back to the forecast if missing.
    for v in ("direct_radiation", "diffuse_radiation"):
        if sat[v] is not None:
            w[v] = sat[v]
    w["radiation_source"] = "satellite" if sat["direct_radiation"] is not None else "forecast"
    w["aod_550nm"] = _hour(AIR_QUALITY, lat, lon, hour, ("aerosol_optical_depth",))["aerosol_optical_depth"]
    return w
