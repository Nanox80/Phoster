"""Phase 3: calibrate the simulator to published natural-dye cells.

Reads data/literature_cells.csv. One row per published cell:
  pigment         chlorophyll | anthocyanin | curcumin  (single-pigment cells only)
  dye_source      plant/extract, e.g. "spinach", "turmeric"
  voc_v           open-circuit voltage (V)
  jsc_ma_cm2      short-circuit current density (mA/cm²)
  ff              fill factor (0-1), leave blank if not reported
  efficiency_pct  efficiency (%)
  citation, doi, notes
Only add cells measured under standard sun (AM1.5G, 100 mW/cm²), which is what the fit assumes.

Fits each pigment's success rate (to its median Jsc) and the shared diode parameters
(to the median Voc and FF), saves cell/calibration.json, then prints the Phase 3 checkpoint:
do simulated single-pigment cells fall inside the published ranges?

Run: python -m cell.calibrate
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from cell.simulator import CALIBRATION_FILE, DEFAULTS, PIGMENTS, reference_sun, simulate

LITERATURE = Path(__file__).parent.parent / "data" / "literature_cells.csv"
METRICS = ("voc_v", "jsc_ma_cm2", "ff", "efficiency_pct")


def simulate_singles(params, pigments=PIGMENTS):
    """Each pure pigment under standard sun (AM1.5G, 25 °C), indexed by pigment."""
    nm, g = reference_sun()
    mixes = np.array([[float(p == q) for q in PIGMENTS] for p in pigments])
    return simulate(nm, g, 1000, 25, mixes, params).set_index(pd.Index(pigments, name="pigment"))


def fit(table):
    """Parameters that best match the median published cell of each pigment."""
    pigments = [p for p in PIGMENTS if p in set(table["pigment"])]
    target = table.groupby("pigment")[["voc_v", "jsc_ma_cm2", "ff"]].median().loc[pigments]

    def params_from(x):
        rates = {**DEFAULTS["success_rate"], **dict(zip(pigments, np.exp(x[:len(pigments)])))}
        return {**DEFAULTS, "success_rate": rates, "i0_ref_a": 10 ** x[-3],
                "r_series_ohm": 10 ** x[-2], "r_shunt_ohm": 10 ** x[-1]}

    def residuals(x):
        # Equal weight: 5% off in current = 50 mV off in Voc = 0.05 off in FF.
        sim = simulate_singles(params_from(x), pigments)
        r = [np.log(sim["jsc_ma_cm2"] / target["jsc_ma_cm2"]) / 0.05, (sim["voc_v"] - target["voc_v"]) / 0.05]
        if target["ff"].notna().any():
            r.append(((sim["ff"] - target["ff"]) / 0.05).dropna())
        return np.concatenate(r)

    d = DEFAULTS
    x0 = [np.log(d["success_rate"][p]) for p in pigments] + [
        np.log10(d["i0_ref_a"]), np.log10(d["r_series_ohm"]), np.log10(d["r_shunt_ohm"])]
    lo = [np.log(1e-3)] * len(pigments) + [-14, -1, 1]
    hi = [0.0] * len(pigments) + [-3, 3, 6]
    return params_from(least_squares(residuals, x0, bounds=(lo, hi)).x)


def checkpoint(table, params):
    """Published min/max vs simulated value for each pigment and metric."""
    sim = simulate_singles(params)
    rows = []
    for pigment, group in table.groupby("pigment"):
        for m in METRICS:
            lo, hi = group[m].min(), group[m].max()
            if pd.isna(lo):
                continue
            v = sim.loc[pigment, m]
            rows.append({"pigment": pigment, "metric": m, "published_min": lo, "published_max": hi,
                         "simulated": round(v, 4), "inside": lo <= v <= hi})
    return pd.DataFrame(rows)


def main():
    table = pd.read_csv(LITERATURE)
    unknown = set(table["pigment"]) - set(PIGMENTS)
    if unknown:
        raise SystemExit(f"Unknown pigment names in {LITERATURE.name}: {unknown}. Use {PIGMENTS}.")
    if table.empty:
        raise SystemExit(f"{LITERATURE} has no rows yet. Add published cells, then rerun.")

    params = fit(table)
    CALIBRATION_FILE.write_text(json.dumps(params, indent=2))
    print(f"Fitted from {len(table)} published cells -> {CALIBRATION_FILE}")
    print(json.dumps(params, indent=2), "\n")
    report = checkpoint(table, params)
    print(report.to_string(index=False))
    print(f"\nCheckpoint: {report['inside'].sum()} of {len(report)} inside published ranges")


if __name__ == "__main__":
    main()
