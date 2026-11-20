"""Test whether the recommender's uncertainty numbers mean what they say.

Without lab data, "reality" is simulated: each true world is the simulator with its inputs and
settings drawn at random. Checks, over real past Malaysian weather:
  1. Calibration   - mixes labelled X% accurate succeed in ~X% of true worlds; 5-95% ranges hold ~90%.
  2. Robustness    - same, when real errors are 2x larger than assumed, or pigments interact.
  3. Sensitivity   - roadmap Phase 8: spectrum ±15% / ±25% bluer or redder; does the top pick hold?
  4. Convergence   - do 500 Monte Carlo runs give stable accuracy numbers?

Run: python -m model.check_uncertainty [--hours 5] [--worlds 200]
"""
import argparse

import numpy as np
import pandas as pd

from cell.generate import LOCATIONS, sample_hours
from cell.simulator import MIXES, simulate
from cell.temperature import cell_temperature
from environment.spectrum import spectrum_from_weather
from model.recommend import BAR, UNCERTAINTY, meets_bar, monte_carlo, rank_mixes

SITES = ["Kuala Lumpur", "Cameron Highlands", "Kota Bharu", "Johor Bahru", "Kuching", "Kota Kinabalu"]
SCENARIOS = {  # how "reality" differs from what the recommender assumes
    "as assumed": {"error_scale": 1, "interaction": 0.0},
    "errors 2x larger": {"error_scale": 2, "interaction": 0.0},
    "unmodelled interactions (±20%)": {"error_scale": 1, "interaction": 0.2},
}
PAIRS = [(0, 1), (0, 2), (1, 2)]
BINS = [0, 10, 30, 50, 70, 90, 99.9, 100]


def conditions(hours_per_site, seed=7):
    """(label, nm, spectrum, sunlight, cell temp) for random past daylight hours."""
    out = []
    for name in SITES:
        lat, lon = LOCATIONS[name]
        for hour, w in sample_hours(lat, lon, "2025-01-01", "2025-12-31", hours_per_site, seed).iterrows():
            when = hour - pd.Timedelta(minutes=30)
            spec = spectrum_from_weather(lat, lon, when, w)
            sun = w["direct_radiation"] + w["diffuse_radiation"]
            out.append((f"{name} {when:%m-%d %H:%M}Z", spec.index.values, spec["total"].values, sun,
                        float(cell_temperature(w["temperature_2m"], sun))))
    return out


def true_worlds(nm, spec, sun, t, n, seed, error_scale, interaction):
    """Efficiency of every mix in `n` simulated realities: (n, 231)."""
    eff = monte_carlo(nm, spec, sun, t, runs=n, seed=seed,
                      uncertainty={k: v * error_scale for k, v in UNCERTAINTY.items()})
    if interaction:
        # Each world gets a random effect per pigment pair, strongest at 50/50, zero for single pigments.
        overlap = np.stack([4 * MIXES[:, i] * MIXES[:, j] for i, j in PAIRS], axis=1)  # (231, 3)
        strength = np.random.default_rng(seed + 1).normal(0, interaction, (n, len(PAIRS)))
        eff = eff * np.exp(strength @ overlap.T)
    return eff


def check_calibration(conds, n_worlds):
    rows, bins = [], []
    for s_name, s in SCENARIOS.items():
        for k, (label, nm, spec, sun, t) in enumerate(conds):
            ranked = rank_mixes(nm, spec, sun, t)
            top, pred = ranked.iloc[0], ranked.sort_values("mix_id")      # top = what the app recommends
            truth = true_worlds(nm, spec, sun, t, n_worlds, seed=1000 + k, **s)
            hit = meets_bar(truth)                                        # (worlds, 231)
            inside = (truth >= pred["efficiency_low"].values) & (truth <= pred["efficiency_high"].values)
            i = int(top["mix_id"])
            rows.append({"scenario": s_name, "condition": label, "range_coverage": inside.mean() * 100,
                         "top_predicted_acc": top["accuracy_pct"], "top_true_pass": hit[:, i].mean() * 100,
                         "top_true_eff_vs_best": (truth[:, i] / truth.max(axis=1)).mean() * 100})
            bins.append(pd.DataFrame({"scenario": s_name, "predicted": pred["accuracy_pct"].values,
                                      "observed": hit.mean(axis=0) * 100}))
    return pd.DataFrame(rows), pd.concat(bins)


def check_sensitivity(conds):
    rows = []
    for label, nm, spec, sun, t in conds:
        position = (nm - 600) / 200
        base = simulate(nm, spec, sun, t)["efficiency_pct"].values
        pick = base.argmax()
        for d in (-0.25, -0.15, 0.15, 0.25):  # negative = bluer
            eff = simulate(nm, spec * np.clip(1 + d * position, 0, None), sun, t)["efficiency_pct"].values
            rows.append({"distortion": f"{'redder' if d > 0 else 'bluer'} {abs(d):.0%}", "condition": label,
                         "pick_changed": eff.argmax() != pick, "old_pick_vs_new_best": eff[pick] / eff.max() * 100})
    return pd.DataFrame(rows)


def check_convergence(cond, seeds=10):
    label, nm, spec, sun, t = cond
    runs = [rank_mixes(nm, spec, sun, t, seed=s) for s in range(seeds)]
    acc = np.array([r.sort_values("mix_id")["accuracy_pct"].values for r in runs])
    tops = {int(r.iloc[0]["mix_id"]) for r in runs}
    return label, acc.std(axis=0).max(), len(tops)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--hours", type=int, default=5, help="random past daylight hours per site")
    a.add_argument("--worlds", type=int, default=200, help="simulated realities per condition")
    args = a.parse_args(argv)
    pd.set_option("display.width", 140)

    conds = conditions(args.hours)
    print(f"{len(conds)} conditions: {len(SITES)} sites x {args.hours} random daylight hours of 2025\n")

    summary, bins = check_calibration(conds, args.worlds)
    print("1-2. CALIBRATION AND ROBUSTNESS")
    print(f"   range_coverage: % of true efficiencies inside the 5-95% range (target 90)")
    print(f"   top pick: predicted accuracy vs % of true worlds where it really reached >= {BAR:.0%} of the best")
    print(summary.groupby("scenario", sort=False)[["range_coverage", "top_predicted_acc", "top_true_pass",
                                                   "top_true_eff_vs_best"]].mean().round(1).to_string(), "\n")
    print("   Reliability (all mixes): predicted accuracy band -> how often it really succeeded")
    bins["band"] = pd.cut(bins["predicted"], BINS, include_lowest=True)
    rel = bins.groupby(["scenario", "band"], sort=False, observed=True).agg(
        mixes=("observed", "size"), predicted=("predicted", "mean"), observed=("observed", "mean")).round(1)
    print(rel.to_string(), "\n")
    worst = summary.sort_values("top_true_pass").head(3)[["scenario", "condition", "top_predicted_acc", "top_true_pass"]]
    print("   Hardest conditions for the top pick:\n" + worst.round(1).to_string(index=False), "\n")

    sens = check_sensitivity(conds)
    print("3. SENSITIVITY (roadmap Phase 8): spectrum distorted, nominal simulator")
    print(sens.groupby("distortion").agg(pick_changed_pct=("pick_changed", lambda x: x.mean() * 100),
                                         old_pick_worst=("old_pick_vs_new_best", "min"),
                                         old_pick_mean=("old_pick_vs_new_best", "mean")).round(1).to_string(), "\n")

    label, max_std, n_tops = check_convergence(conds[0])
    print(f"4. CONVERGENCE ({label}, 10 different seeds x 500 runs)")
    print(f"   largest spread in any mix's accuracy: ±{max_std:.1f} points; distinct top picks across seeds: {n_tops}")


if __name__ == "__main__":
    main()
