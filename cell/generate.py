"""Phase 3: generate the simulated training dataset.

Past Open-Meteo weather for Malaysian locations -> Phase 1 spectrum -> Phase 2 cell temperature
-> all 231 mixes through the simulator. Saves data/sim/sim_dataset.csv, every row source = sim.

Run: python -m cell.generate [--hours 50] [--start 2025-01-01 --end 2025-12-31] [--seed 0]
"""
import argparse
from pathlib import Path

import pandas as pd

from cell.simulator import simulate
from cell.temperature import cell_temperature
from environment.spectrum import AS7341_BANDS, colour_bands, spectrum_from_weather, sun_position
from environment.weather import fetch_weather_history

OUT = Path(__file__).parent.parent / "data" / "sim" / "sim_dataset.csv"

# Spread across the peninsula, Sarawak and Sabah, coast and highland.
LOCATIONS = {
    "Kuala Lumpur": (3.139, 101.687), "George Town": (5.414, 100.329), "Alor Setar": (6.121, 100.367),
    "Ipoh": (4.597, 101.090), "Cameron Highlands": (4.471, 101.377), "Kota Bharu": (6.125, 102.238),
    "Kuantan": (3.817, 103.326), "Melaka": (2.189, 102.250), "Johor Bahru": (1.492, 103.741),
    "Kuching": (1.553, 110.359), "Miri": (4.399, 113.991), "Kota Kinabalu": (5.980, 116.073),
    "Sandakan": (5.840, 118.117),
}
NEEDED = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "direct_radiation",
          "diffuse_radiation", "aod_550nm"]


def sample_hours(lat, lon, start, end, n, seed):
    """`n` random daylight hours of past weather at one location."""
    w = fetch_weather_history(lat, lon, start, end).dropna(subset=NEEDED)
    # Radiation is the mean over the hour ending at the timestamp, so model the middle of that hour.
    mid = w.index - pd.Timedelta(minutes=30)
    sunny = (sun_position(lat, lon, mid)["apparent_elevation"].values > 0) & (
        w["direct_radiation"] + w["diffuse_radiation"] > 0).values
    w = w[sunny]
    return w.sample(min(n, len(w)), random_state=seed)


def rows_for_hour(name, lat, lon, hour, weather):
    when = hour - pd.Timedelta(minutes=30)
    spec = spectrum_from_weather(lat, lon, when, weather)
    sunlight = weather["direct_radiation"] + weather["diffuse_radiation"]
    t_cell = cell_temperature(weather["temperature_2m"], sunlight)
    cells = simulate(spec.index.values, spec["total"].values, sunlight, t_cell)
    conditions = {"source": "sim", "location": name, "lat": lat, "lon": lon, "time_utc": when,
                  "air_temp_c": weather["temperature_2m"], "cell_temp_c": t_cell,
                  "cloud_cover_pct": weather["cloud_cover"], "sunlight_w_m2": sunlight,
                  "direct_w_m2": weather["direct_radiation"], "diffuse_w_m2": weather["diffuse_radiation"],
                  **colour_bands(spec, AS7341_BANDS).to_dict()}
    return cells.assign(**conditions)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--hours", type=int, default=50, help="daylight hours sampled per location")
    a.add_argument("--start", default="2025-01-01")
    a.add_argument("--end", default="2025-12-31")
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args(argv)

    frames = []
    for name, (lat, lon) in LOCATIONS.items():
        hours = sample_hours(lat, lon, args.start, args.end, args.hours, args.seed)
        print(f"{name}: {len(hours)} hours")
        frames += [rows_for_hour(name, lat, lon, h, w) for h, w in hours.iterrows()]

    data = pd.concat(frames, ignore_index=True)
    first = ["source", "location", "lat", "lon", "time_utc"]
    data = data[first + [c for c in data.columns if c not in first]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data)} rows ({len(frames)} conditions x 231 mixes) -> {OUT}")


if __name__ == "__main__":
    main()
