"""Offline checks for Phase 1 (no network). Run: .venv/bin/python -m unittest tests/test_environment.py"""
import unittest

import numpy as np
import pandas as pd

from environment.spectrum import colour_bands, ground_spectrum, in_malaysia, spectrum_from_weather

KL = (3.14, 101.69)
NOON_KL = pd.Timestamp("2026-11-19 05:00", tz="UTC")  # 13:00 local
WEATHER = {"temperature_2m": 32, "relative_humidity_2m": 65, "surface_pressure": 1007,
           "total_column_integrated_water_vapour": 50, "aod_550nm": 0.3}


def spectrum(direct, diffuse):
    return spectrum_from_weather(*KL, NOON_KL, {**WEATHER, "direct_radiation": direct,
                                               "diffuse_radiation": diffuse})


def blue_red_ratio(df):
    return df.loc[400:500, "total"].sum() / df.loc[600:700, "total"].sum()


class EnvironmentTest(unittest.TestCase):
    def test_malaysia_check(self):
        self.assertTrue(in_malaysia(*KL))
        self.assertTrue(in_malaysia(5.98, 116.07))   # Kota Kinabalu
        self.assertFalse(in_malaysia(13.75, 100.5))  # Bangkok
        with self.assertRaises(ValueError):
            ground_spectrum(51.5, -0.1)

    def test_night_offers_next_daylight(self):
        out = ground_spectrum(*KL, pd.Timestamp("2026-11-19 16:00", tz="UTC"))  # midnight local
        self.assertEqual(out["status"], "night")
        # sunrise ~07:05 local, so the first whole sunlit hour is 08:00 local
        self.assertEqual(out["next_daylight"], pd.Timestamp("2026-11-20 00:00", tz="UTC"))

    def test_spectrum_shape_and_units(self):
        df = spectrum(600, 250)
        self.assertGreaterEqual(df.index.min(), 400)
        self.assertLessEqual(df.index.max(), 800)
        self.assertTrue((df >= 0).all().all())
        self.assertTrue(0.5 < df["total"].max() < 2.5)  # W/m²/nm, plausible midday peak

    def test_cloudy_sky_is_bluer(self):
        self.assertGreater(blue_red_ratio(spectrum(50, 300)), blue_red_ratio(spectrum(700, 150)))

    def test_colour_bands_cover_whole_range(self):
        df = spectrum(600, 250)
        bands = colour_bands(df)
        self.assertTrue((bands > 0).all())  # yellow too, despite no SPECTRL2 point inside it
        whole = np.trapezoid(np.interp(np.arange(400, 801), df.index, df["total"]))
        self.assertAlmostEqual(bands.sum(), whole, places=6)

    def test_no_light_gives_zero_spectrum(self):
        self.assertTrue(np.allclose(spectrum(0, 0)["total"], 0))


if __name__ == "__main__":
    unittest.main()
