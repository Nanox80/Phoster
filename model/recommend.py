"""Phase 5: rank all 231 pigment mixes and give each an honest prediction accuracy.

Processing is the physics simulator itself (cell/simulator.py), not an ML copy of it.
Prediction accuracy = % of Monte Carlo runs in which a mix reaches >= 85% of that run's best mix,
each run nudging the inputs within their uncertainties.

After the Phase 7 experiment, the UNCERTAINTY values come from your measurements and an ML
correction is added on top (see CALIBRATION_STEPS.txt, Step 10).
"""
import numpy as np
import pandas as pd

from cell.simulator import CALIBRATION_FILE, MIXES, PIGMENTS, load_params, simulate
from cell.temperature import cell_temperature
from environment.spectrum import LOCAL_TZ, colour_bands, ground_spectrum

RUNS = 500
SEED = 0              # fixed, so the same inputs always give the same ranking
BAR = 0.85            # success target: >= 85% of the best mix
WARN_BELOW = 60       # warn if the top pick's accuracy (%) is under this
SIMILAR_PCT = 10      # mixes within 10 percentage points of a shown mix are merged into it
TIE_PTS = 5           # accuracy within 5 points of the top = tied (500-run noise measured at ±3.8)

# 1-sigma uncertainties. Placeholders until Phase 7 measures them (CALIBRATION_STEPS.txt Steps 6, 8, 10).
UNCERTAINTY = {
    "sunlight": 0.10,      # relative error in total light
    "colour_tilt": 0.10,   # spectrum bluer/redder: ±10% at 400/800 nm, 0 at 600 nm
    "cell_temp_c": 3.0,    # °C
    "success_rate": 0.15,  # relative, per pigment (how sure we are of each pigment's calibration)
}
# Cell-to-cell scatter is deliberately NOT added per mix: the target compares average cells, and
# independent noise on 231 mixes inflates each run's "best" (winner's curse), making every mix look worse.
# Mix-specific model error enters after Phase 7 through the ML correction's own uncertainty.


def monte_carlo(nm, spectrum, sunlight, cell_temp_c, params=None, runs=RUNS, seed=SEED,
                uncertainty=UNCERTAINTY):
    """Efficiency (%) of every mix in every run: array (runs, 231)."""
    p = params or load_params()
    u = uncertainty
    rng = np.random.default_rng(seed)
    position = (np.asarray(nm) - 600) / 200  # -1 at 400 nm, +1 at 800 nm
    eff = np.empty((runs, len(MIXES)))
    for r in range(runs):
        scale = max(1 + rng.normal(0, u["sunlight"]), 0.05)
        tilt = np.clip(1 + rng.normal(0, u["colour_tilt"]) * position, 0, None)
        rates = {k: v * np.exp(rng.normal(0, u["success_rate"])) for k, v in p["success_rate"].items()}
        sim = simulate(nm, spectrum * scale * tilt, sunlight * scale,
                       cell_temp_c + rng.normal(0, u["cell_temp_c"]), params={**p, "success_rate": rates})
        eff[r] = sim["efficiency_pct"].values
    return eff


def meets_bar(eff):
    """True where a mix reaches >= 85% of the best mix in the same run/world. eff: (runs, 231)."""
    return eff >= BAR * eff.max(axis=1, keepdims=True)


def rank_mixes(nm, spectrum, sunlight, cell_temp_c, params=None, **mc):
    """All 231 mixes, best first, with predicted efficiency and prediction accuracy (%)."""
    df = simulate(nm, spectrum, sunlight, cell_temp_c, params=params)
    eff = monte_carlo(nm, spectrum, sunlight, cell_temp_c, params, **mc)
    df["mix_id"] = np.arange(len(MIXES))  # row of MIXES, so results can be matched back
    df["accuracy_pct"] = (meets_bar(eff).mean(axis=0) * 100).round(1)
    df["efficiency_low"], df["efficiency_high"] = np.percentile(eff, [5, 95], axis=0)
    df["n_pigments"] = (MIXES > 0).sum(axis=1)
    # Accuracy differences smaller than the Monte Carlo noise are not real: treat every mix within
    # TIE_PTS of the top as tied best ("plateau") and order those by predicted efficiency instead,
    # so the top pick no longer depends on the random seed.
    df["tied_best"] = df["accuracy_pct"] >= df["accuracy_pct"].max() - TIE_PTS
    plateau = df[df["tied_best"]].sort_values("efficiency_pct", ascending=False)
    rest = df[~df["tied_best"]].sort_values(["accuracy_pct", "efficiency_pct"], ascending=False)
    return pd.concat([plateau, rest], ignore_index=True)


def distinct(ranked, n=10, similar=SIMILAR_PCT):
    """Top `n` mixes with near-duplicates merged; `similar_mixes` counts what each absorbed."""
    pct = ranked[[f"pct_{p}" for p in PIGMENTS]].values
    shown, merged = [], []
    for i in range(len(ranked)):
        near = [j for j, s in enumerate(shown) if np.abs(pct[i] - pct[s]).max() <= similar]
        if near:
            merged[near[0]] += 1
        elif len(shown) < n:
            shown.append(i)
            merged.append(0)
    return ranked.iloc[shown].assign(similar_mixes=merged)


def picks(ranked):
    """Best overall, best 2-pigment, best single pigment (always shown, per the roadmap)."""
    return {"best_overall": ranked.iloc[0],
            "best_two_pigment": ranked[ranked["n_pigments"] == 2].iloc[0],
            "best_single": ranked[ranked["n_pigments"] == 1].iloc[0]}


def recommend_for_location(lat, lon, when=None, **mc):
    """Whole pipeline for a coordinate: live weather -> spectrum -> cell temperature -> ranking.

    Returns a JSON-ready dict. status is "ok" or "night". Raises ValueError outside Malaysia.
    """
    env = ground_spectrum(lat, lon, when)
    if env["status"] == "night":
        return {"status": "night", "lat": lat, "lon": lon,
                "next_daylight_utc": env["next_daylight"].isoformat(),
                "next_daylight_local": f"{env['next_daylight'].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M}"}

    w, spec = env["weather"], env["spectrum"]
    sunlight = w["direct_radiation"] + w["diffuse_radiation"]
    t_cell = float(cell_temperature(w["temperature_2m"], sunlight))
    ranked = rank_mixes(spec.index.values, spec["total"].values, sunlight, t_cell, **mc)
    top = picks(ranked)

    def mix(row):
        def cast(k, v):
            if k == "tied_best":
                return bool(v)
            return int(v) if k.startswith("pct_") or k in ("n_pigments", "similar_mixes", "mix_id") else float(v)
        return {k: cast(k, v) for k, v in row.items()}

    tied = ranked[ranked["tied_best"]]
    plateau = {"size": len(tied), "tie_points": TIE_PTS,
               "ranges": {p: [int(tied[f"pct_{p}"].min()), int(tied[f"pct_{p}"].max())] for p in PIGMENTS}}

    warning = None
    if top["best_overall"]["accuracy_pct"] < WARN_BELOW:
        warning = (f"Low confidence: the top pick reaches 85% of the best mix in only "
                   f"{top['best_overall']['accuracy_pct']:.0f}% of runs. Several mixes are nearly tied "
                   f"or conditions are uncertain (e.g. patchy cloud, low sun).")
    return {
        "status": "ok", "lat": lat, "lon": lon,
        "time_local": f"{env['time'].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M}",
        "conditions": {"air_temp_c": w["temperature_2m"], "cell_temp_c": round(t_cell, 1),
                       "cloud_cover_pct": w["cloud_cover"], "haze_aod": w["aod_550nm"],
                       "sun_elevation_deg": round(float(env["sun_elevation"]), 1),
                       "direct_w_m2": w["direct_radiation"], "diffuse_w_m2": w["diffuse_radiation"],
                       "radiation_source": w["radiation_source"]},
        "spectrum": {"nm": spec.index.tolist(), "direct": spec["direct"].tolist(),
                     "diffuse": spec["diffuse"].tolist(), "total": spec["total"].tolist()},
        "colour_bands_w_m2": colour_bands(spec).round(2).to_dict(),
        "picks": {k: mix(v) for k, v in top.items()},
        "plateau": plateau,
        "ranked": [mix(r) for _, r in distinct(ranked).iterrows()],
        "all_mixes": [mix(r) for _, r in ranked.iterrows()],
        "warning": warning,
        "runs": mc.get("runs", RUNS),
        "calibrated": CALIBRATION_FILE.exists(),
    }
