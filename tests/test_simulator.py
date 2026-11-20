"""Phase 3 checks (offline). Run: .venv/bin/python -m unittest tests/test_simulator.py"""
import unittest

import numpy as np
import pandas as pd

from cell.calibrate import checkpoint, fit, simulate_singles
from cell.simulator import DEFAULTS, MIXES, PIGMENTS, absorption, reference_sun, simulate

NM, SUN = reference_sun()
SINGLES = np.eye(3)


class SimulatorTest(unittest.TestCase):
    def test_231_mixes_in_5pct_steps(self):
        self.assertEqual(len(MIXES), 231)
        self.assertTrue(np.allclose(MIXES.sum(1), 1))
        self.assertEqual(len({tuple(m) for m in MIXES}), 231)

    def test_absorption_peaks(self):
        nm = np.arange(400, 801)
        a = absorption(nm)
        self.assertTrue(np.allclose(a.max(1), 1, atol=0.01))
        peaks = dict(zip(PIGMENTS, nm[a.argmax(1)]))
        self.assertAlmostEqual(peaks["chlorophyll"], 666, delta=3)
        self.assertAlmostEqual(peaks["anthocyanin"], 522, delta=6)
        self.assertAlmostEqual(peaks["curcumin"], 425, delta=3)

    def test_no_light_no_power(self):
        df = simulate(NM, SUN * 0, 0, 25, SINGLES, DEFAULTS)
        self.assertTrue(np.allclose(df[["jsc_ma_cm2", "pmax_mw_cm2", "efficiency_pct"]], 0))

    def test_current_scales_with_light(self):
        full = simulate(NM, SUN, 1000, 25, SINGLES, DEFAULTS)["jsc_ma_cm2"]
        half = simulate(NM, SUN / 2, 500, 25, SINGLES, DEFAULTS)["jsc_ma_cm2"]
        self.assertTrue(np.allclose(half / full, 0.5, atol=0.01))

    def test_hotter_cell_lower_voltage(self):
        cool = simulate(NM, SUN, 1000, 25, SINGLES, DEFAULTS)["voc_v"]
        hot = simulate(NM, SUN, 1000, 45, SINGLES, DEFAULTS)["voc_v"]
        self.assertTrue((hot < cool).all())

    def test_red_light_favours_chlorophyll(self):
        red = np.where(np.abs(NM - 666) < 5, 1.0, 0.0)  # narrow red LED
        jsc = simulate(NM, red, 10, 25, SINGLES, DEFAULTS).set_index(pd.Index(PIGMENTS))["jsc_ma_cm2"]
        self.assertGreater(jsc["chlorophyll"], 10 * jsc["curcumin"])

    def test_plausible_natural_dye_cell(self):
        df = simulate(NM, SUN, 1000, 25, SINGLES, DEFAULTS)
        self.assertTrue(df["voc_v"].between(0.3, 0.7).all())
        self.assertTrue(df["efficiency_pct"].between(0.01, 2).all())


class CalibrationTest(unittest.TestCase):
    # Made-up test cells, NOT published data.
    TABLE = pd.DataFrame({
        "pigment": ["chlorophyll"] * 2 + ["anthocyanin"] * 2 + ["curcumin"] * 2,
        "voc_v": [0.50, 0.56, 0.52, 0.58, 0.48, 0.54],
        "jsc_ma_cm2": [0.8, 1.2, 1.5, 2.1, 0.4, 0.6],
        "ff": [0.50, 0.60, 0.50, 0.60, 0.50, 0.60],
        "efficiency_pct": [0.2, 0.4, 0.4, 0.7, 0.1, 0.2],
    })

    def test_fit_lands_inside_ranges(self):
        params = fit(self.TABLE)
        sim = simulate_singles(params)
        medians = self.TABLE.groupby("pigment")["jsc_ma_cm2"].median()
        for p in PIGMENTS:
            self.assertAlmostEqual(sim.loc[p, "jsc_ma_cm2"], medians[p], delta=0.05 * medians[p])
        report = checkpoint(self.TABLE, params)
        voc = report[report.metric.isin(["voc_v", "jsc_ma_cm2"])]
        self.assertTrue(voc["inside"].all(), voc)


if __name__ == "__main__":
    unittest.main()
