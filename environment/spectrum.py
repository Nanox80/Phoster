"""Phase 1: coordinates + time -> spectrum of sunlight on the ground (horizontal surface).

ground_spectrum(lat, lon) checks the input, fetches live weather, runs SPECTRL2 and
scales it to the real direct/diffuse sunlight. Step 5 (AS7341 correction) needs
sensor readings and is not built yet.
"""
import numpy as np
import pandas as pd
import pvlib

from cell.temperature import cell_temperature
from environment.weather import fetch_weather

# Typical tropical values (placeholders; refine if better local data is found).
OZONE_ATM_CM = 0.26   # ~260 Dobson units, typical total-column ozone near the equator
GROUND_ALBEDO = 0.2   # vegetation / mixed urban ground
KEEP_NM = (400, 800)  # band the pigments care about

# shortcut: two bounding boxes, so border strips of southern Thailand, Singapore, Brunei
# and Kalimantan pass too; swap for a real boundary polygon if that matters.
MALAYSIA_BOXES = (
    (1.2, 6.8, 99.6, 104.6),    # Peninsular: lat min, lat max, lon min, lon max
    (0.85, 7.4, 109.5, 119.3),  # Sabah + Sarawak
)


def in_malaysia(lat, lon):
    return any(a <= lat <= b and c <= lon <= d for a, b, c, d in MALAYSIA_BOXES)


def sun_position(lat, lon, times):
    return pvlib.solarposition.get_solarposition(pd.DatetimeIndex(times), lat, lon)


def next_daylight_hour(lat, lon, when):
    """First whole hour after `when` with the sun above the horizon."""
    hours = pd.date_range(when.ceil("h"), periods=25, freq="h")
    up = sun_position(lat, lon, hours)["apparent_elevation"] > 0
    return up.idxmax()


def spectrum_from_weather(lat, lon, when, weather):
    """SPECTRL2 clear-sky spectrum scaled to the real direct and diffuse sunlight.

    Returns a DataFrame indexed by wavelength (nm) with columns direct, diffuse, total
    in W/m²/nm, on a horizontal surface, cropped to KEEP_NM.
    """
    sun = sun_position(lat, lon, [when]).iloc[0]
    zenith = sun["apparent_zenith"]
    airmass = pvlib.atmosphere.get_relative_airmass(zenith, model="kastenyoung1989")

    pw = weather.get("total_column_integrated_water_vapour")
    pw_cm = pw / 10 if pd.notna(pw) else float(
        pvlib.atmosphere.gueymard94_pw(weather["temperature_2m"], weather["relative_humidity_2m"]))
    aod_500 = pvlib.atmosphere.angstrom_aod_at_lambda(weather["aod_550nm"], 550, lambda1=500)

    s = pvlib.spectrum.spectrl2(
        apparent_zenith=zenith, aoi=zenith, surface_tilt=0, ground_albedo=GROUND_ALBEDO,
        surface_pressure=weather["surface_pressure"] * 100,  # hPa -> Pa
        relative_airmass=airmass, precipitable_water=pw_cm, ozone=OZONE_ATM_CM,
        aerosol_turbidity_500nm=aod_500, dayofyear=when.day_of_year,
    )
    wl = s["wavelength"]
    direct = s["poa_direct"].ravel()
    diffuse = s["poa_sky_diffuse"].ravel()

    # Step 4: scale each part so its full-range total matches the measured W/m².
    # Cloudy skies -> more diffuse (bluer) light, so the colour shifts automatically.
    def scale(spec, measured):
        clear = np.trapezoid(spec, wl)
        return spec * (measured / clear) if clear > 0 else spec * 0

    df = pd.DataFrame({
        "direct": scale(direct, weather["direct_radiation"]),
        "diffuse": scale(diffuse, weather["diffuse_radiation"]),
    }, index=pd.Index(wl, name="wavelength_nm"))
    df["total"] = df["direct"] + df["diffuse"]
    return df.loc[KEEP_NM[0]:KEEP_NM[1]]


def ground_spectrum(lat, lon, when=None):
    """Main entry point. `when` defaults to now (UTC).

    Returns {"status": "night", "next_daylight": Timestamp}
         or {"status": "ok", "time", "weather", "sun_elevation", "spectrum": DataFrame}.
    Raises ValueError for coordinates outside Malaysia.
    """
    if not in_malaysia(lat, lon):
        raise ValueError(f"({lat}, {lon}) is outside Malaysia")
    when = pd.Timestamp.now(tz="UTC") if when is None else pd.Timestamp(when).tz_convert("UTC")

    elevation = sun_position(lat, lon, [when])["apparent_elevation"].iloc[0]
    if elevation <= 0:
        return {"status": "night", "next_daylight": next_daylight_hour(lat, lon, when)}

    # Open-Meteo radiation is the mean over the hour ending at the timestamp,
    # so the hour that contains `when` is labelled by the next full hour.
    weather = fetch_weather(lat, lon, when.ceil("h"))
    return {"status": "ok", "time": when, "weather": weather, "sun_elevation": elevation,
            "spectrum": spectrum_from_weather(lat, lon, when, weather)}


COLOURS = {"violet": (400, 450), "blue": (450, 495), "green": (495, 570), "yellow": (570, 590),
           "orange": (590, 620), "red": (620, 750), "near-IR": (750, 800)}
# AS7341 channels F1-F8 as centre ± half the datasheet FWHM (nm).
# shortcut: real channel responses are bell-shaped; use the datasheet curves once the sensor is calibrated (Phase 1 Step 5).
AS7341_BANDS = {f"F{i}_{c}nm": (c - w / 2, c + w / 2) for i, (c, w) in enumerate(
    [(415, 26), (445, 30), (480, 36), (515, 39), (555, 39), (590, 40), (630, 50), (680, 52)], start=1)}
LOCAL_TZ = "Asia/Kuala_Lumpur"


def colour_bands(df, bands=COLOURS):
    """W/m² of light in each band {name: (lo_nm, hi_nm)}."""
    # SPECTRL2 points are 10-25 nm apart (none inside 570-590), so interpolate finely first.
    out = {}
    for name, (lo, hi) in bands.items():
        nm = np.linspace(lo, hi, int(hi - lo) + 1)
        out[name] = np.trapezoid(np.interp(nm, df.index.values, df["total"].values), nm)
    return pd.Series(out)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Live sunlight spectrum at a Malaysian location.")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--plot", action="store_true", help="show a graph of the spectrum")
    a = p.parse_args(argv)

    try:
        out = ground_spectrum(a.lat, a.lon)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")
    if out["status"] == "night":
        nxt = out["next_daylight"].tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        raise SystemExit(f"No sunlight right now. Next daylight hour: {nxt} (Malaysia time)")

    w, df = out["weather"], out["spectrum"]
    bands = colour_bands(df)
    print(f"Location: {a.lat}, {a.lon}  |  {out['time'].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M} Malaysia time")
    print(f"Sun {out['sun_elevation']:.0f}° up  |  {w['temperature_2m']} °C air  |  "
          f"{w['cloud_cover']}% cloud  |  haze (AOD) {w['aod_550nm']}")
    sunlight = w["direct_radiation"] + w["diffuse_radiation"]
    print(f"Sunlight: direct {w['direct_radiation']:.0f} W/m², diffuse {w['diffuse_radiation']:.0f} W/m² "
          f"({w['radiation_source']})")
    print(f"Estimated cell temperature: {cell_temperature(w['temperature_2m'], sunlight):.1f} °C\n")
    print(f"{'Colour':<8} {'nm':>9} {'W/m²':>7} {'share':>6}")
    for name, value in bands.items():
        lo, hi = COLOURS[name]
        print(f"{name:<8} {lo:>4}-{hi:<4} {value:>7.1f} {value / bands.sum():>6.0%}")
    print(f"{'total':<8} {'400-800':>9} {bands.sum():>7.1f}")

    if a.plot:
        import matplotlib.pyplot as plt
        ax = df.plot(title=f"Sunlight spectrum at {a.lat}, {a.lon}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("W/m²/nm")
        plt.show()


if __name__ == "__main__":
    main()
