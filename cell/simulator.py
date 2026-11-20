"""Phase 3: DSSC physics simulator.

Chain: light spectrum -> light each pigment absorbs (Beer-Lambert mix) -> photocurrent
(per-pigment success rate) -> single-diode equation at the cell temperature -> Voc, Isc, Pmax, efficiency.
Assumes no pigment interactions in mixes (checked against real mixed cells in Phase 7).
"""
import functools
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

PIGMENTS = ("chlorophyll", "anthocyanin", "curcumin")
SPECTRA_DIR = Path(__file__).parent / "spectra"
SPECTRA_FILES = {"chlorophyll": "chlorophyll_a.txt", "anthocyanin": "anthocyanin_c3g.txt",
                 "curcumin": "curcumin.txt"}
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"

# All 231 mixes in 5% steps, as fractions summing to 1: columns = PIGMENTS.
MIXES = np.array([(a, b, 20 - a - b) for a, b in itertools.product(range(21), repeat=2) if a + b <= 20]) / 20

# Placeholder parameters for a 1 cm² natural-dye cell (typical textbook values), replaced by
# `python -m cell.calibrate` once data/literature_cells.csv has rows, and by your own cells in Phase 7.
DEFAULTS = {
    "success_rate": {p: 0.2 for p in PIGMENTS},  # fraction of absorbed photons that become current
    "peak_absorbance": 1.0,      # film absorbance at each pigment's peak ("same strength")
    "area_cm2": 1.0,
    "i0_ref_a": 1e-7,            # diode saturation current at 25 °C
    "ideality": 2.0,
    "r_series_ohm": 30.0,
    "r_shunt_ohm": 2000.0,
    "eg_ev": 0.55,               # gives Voc ≈ -2.5 mV/°C (typical DSSC); refit from Phase 7 two-temperature tests
}

Q = 1.602176634e-19
H = 6.62607015e-34
C = 2.99792458e8
K_B = 1.380649e-23
K_B_EV = 8.617333262e-5


def load_params():
    """Calibrated parameters if `cell/calibration.json` exists, else DEFAULTS."""
    if CALIBRATION_FILE.exists():
        return {**DEFAULTS, **json.loads(CALIBRATION_FILE.read_text())}
    return DEFAULTS


def absorption(nm):
    """Each pigment's absorption on wavelengths `nm`, scaled so its peak in 400-800 nm is 1.

    Returns array (3, len(nm)); zero outside each file's measured range.
    """
    return _absorption(tuple(np.asarray(nm, dtype=float)))


@functools.lru_cache(maxsize=8)  # Monte Carlo calls this hundreds of times on the same wavelengths
def _absorption(nm):
    nm = np.array(nm)
    rows = []
    for p in PIGMENTS:
        data = np.loadtxt(SPECTRA_DIR / SPECTRA_FILES[p], skiprows=1)
        visible = (data[:, 0] >= 400) & (data[:, 0] <= 800)
        rows.append(np.interp(nm, data[:, 0], data[:, 1] / data[visible, 1].max(), left=0, right=0))
    return np.array(rows)


def photocurrent(nm, spectral_irradiance, mixes=MIXES, params=None):
    """Photocurrent (A) of each mix under a spectrum (W/m²/nm on wavelengths `nm`)."""
    p = params or load_params()
    a = p["peak_absorbance"] * absorption(nm)          # (3, λ)
    a_mix = mixes @ a                                   # (mixes, λ) Beer-Lambert: absorbances add
    absorbed = 1 - 10.0 ** -a_mix                       # fraction of light the film absorbs
    share = np.divide(mixes[:, :, None] * a, a_mix[:, None, :],
                      out=np.zeros((len(mixes), *a.shape)), where=a_mix[:, None, :] > 0)
    eta = np.array([p["success_rate"][k] for k in PIGMENTS])
    useful = absorbed * np.einsum("p,mpl->ml", eta, share)
    photons = spectral_irradiance * (nm * 1e-9) / (H * C)   # photons /s /m² /nm
    return Q * np.trapezoid(useful * photons, nm, axis=1) * p["area_cm2"] * 1e-4


def simulate(nm, spectral_irradiance, irradiance_w_m2, cell_temp_c, mixes=MIXES, params=None):
    """Simulate every mix. Returns a DataFrame with one row per mix.

    irradiance_w_m2 is the total sunlight on the cell (all wavelengths), used for efficiency.
    """
    p = params or load_params()
    t = cell_temp_c + 273.15
    t_ref = 298.15
    i0 = p["i0_ref_a"] * (t / t_ref) ** 3 * np.exp(p["eg_ev"] / K_B_EV * (1 / t_ref - 1 / t))
    n_vth = p["ideality"] * K_B * t / Q
    il = photocurrent(nm, np.asarray(spectral_irradiance), mixes, p)
    out = pvlib.pvsystem.singlediode(il, i0, p["r_series_ohm"], p["r_shunt_ohm"], n_vth)

    area_m2 = p["area_cm2"] * 1e-4
    df = pd.DataFrame(mixes * 100, columns=[f"pct_{k}" for k in PIGMENTS]).round().astype(int)
    df["jsc_ma_cm2"] = out["i_sc"] * 1e3 / p["area_cm2"]
    df["voc_v"] = out["v_oc"]
    df["pmax_mw_cm2"] = out["p_mp"] * 1e3 / p["area_cm2"]
    df["ff"] = np.divide(out["p_mp"], out["i_sc"] * out["v_oc"],
                         out=np.zeros(len(mixes)), where=out["i_sc"] * out["v_oc"] > 0)
    df["efficiency_pct"] = (out["p_mp"] / (irradiance_w_m2 * area_m2) * 100 if irradiance_w_m2 > 0
                            else np.zeros(len(mixes)))
    return df


def reference_sun():
    """ASTM G173 AM1.5G spectrum (1000 W/m²) - the standard light published cells are tested under."""
    ref = pvlib.spectrum.get_reference_spectra()
    return ref.index.values, ref["global"].values
