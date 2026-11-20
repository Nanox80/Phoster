"""Phase 6 v1: command-line recommendation.

Run: python -m app.predict --lat 3.14 --lon 101.69
"""
import argparse

from model.recommend import recommend_for_location


def main(argv=None):
    p = argparse.ArgumentParser(description="Best DSSC pigment mixes right now at a Malaysian location.")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    a = p.parse_args(argv)
    try:
        r = recommend_for_location(a.lat, a.lon)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")
    if r["status"] == "night":
        raise SystemExit(f"No sunlight right now. Next daylight hour: {r['next_daylight_local']} (Malaysia time)")

    c = r["conditions"]
    print(f"Location: {a.lat}, {a.lon}  |  {r['time_local']}  |  {c['cloud_cover_pct']:.0f}% cloud, "
          f"{c['air_temp_c']} °C air, cell {c['cell_temp_c']} °C")
    if not r["calibrated"]:
        print("(Simulator uses placeholder settings until calibrated - see CALIBRATION_STEPS.txt)")
    pl = r["plateau"]
    if pl["size"] > 1:
        rng = ", ".join(f"{name} {lo}-{hi}%" for name, (lo, hi) in
                        zip(("Chl", "Anth", "Curc"), pl["ranges"].values()))
        print(f"{pl['size']} mixes are practically tied for best ({rng}); any of them is a good choice. "
              f"Marked * below, ordered by predicted efficiency.")
    print(f"{'Rank':<5}{'Chl':>5}{'Anth':>6}{'Curc':>6}   {'Pred. efficiency':<17}{'Accuracy (>=85% of best)'}")
    for i, m in enumerate(r["ranked"], 1):
        print(f"{i:<5}{m['pct_chlorophyll']:>4}%{m['pct_anthocyanin']:>5}%{m['pct_curcumin']:>5}%   "
              f"{m['efficiency_pct']:.3f}%{'':<11}{m['accuracy_pct']:.0f}%{' *' if m['tied_best'] else ''}")
    if r["warning"]:
        print(f"\nWARNING: {r['warning']}")


if __name__ == "__main__":
    main()
