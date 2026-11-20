"""Phase 5/6 checks (offline). Run: .venv/bin/python -m unittest tests/test_recommend.py"""
import unittest
from http import HTTPStatus

import numpy as np

from app.web import api_recommend
from cell.simulator import reference_sun
from model.recommend import UNCERTAINTY, distinct, picks, rank_mixes

nm, g = reference_sun()
keep = (nm >= 400) & (nm <= 800)
NM, SUN = nm[keep], g[keep]


def ranked(**mc):
    return rank_mixes(NM, SUN, 1000, 40, runs=100, **mc)


class RecommendTest(unittest.TestCase):
    def test_same_inputs_same_ranking(self):  # roadmap Phase 5 checkpoint, part 1
        a, b = ranked(), ranked()
        self.assertTrue(a.equals(b))

    def test_noisier_inputs_lower_accuracy(self):  # roadmap Phase 5 checkpoint, part 2
        noisy = {k: v * 3 for k, v in UNCERTAINTY.items()}
        calm = ranked()["accuracy_pct"]
        rough = ranked(uncertainty=noisy)["accuracy_pct"]
        self.assertLess(rough.iloc[0], calm.iloc[0] + 1e-9)
        self.assertGreater((calm > 50).sum(), 0)
        self.assertTrue(calm.between(0, 100).all())

    def test_ranked_best_first(self):
        r = ranked()
        self.assertEqual(len(r), 231)
        tied, rest = r[r["tied_best"]], r[~r["tied_best"]]
        self.assertTrue(r["tied_best"].iloc[:len(tied)].all())                 # tied mixes come first,
        self.assertTrue((np.diff(tied["efficiency_pct"]) <= 0).all())          # by efficiency,
        self.assertTrue((np.diff(rest["accuracy_pct"]) <= 0).all())            # then the rest by accuracy
        self.assertTrue((tied["accuracy_pct"] >= r["accuracy_pct"].max() - 5).all())

    def test_top_pick_does_not_depend_on_seed(self):  # fix for the convergence check
        tops = {int(rank_mixes(NM, SUN, 1000, 40, runs=300, seed=s).iloc[0]["mix_id"]) for s in range(5)}
        self.assertEqual(len(tops), 1)

    def test_picks_have_right_pigment_counts(self):
        p = picks(ranked())
        self.assertEqual(p["best_two_pigment"]["n_pigments"], 2)
        self.assertEqual(p["best_single"]["n_pigments"], 1)

    def test_distinct_merges_near_duplicates(self):
        d = distinct(ranked(), n=10)
        pct = d[["pct_chlorophyll", "pct_anthocyanin", "pct_curcumin"]].values
        for i in range(len(pct)):
            for j in range(i):
                self.assertGreater(np.abs(pct[i] - pct[j]).max(), 10)


    def test_uncertainty_is_calibrated_when_assumptions_hold(self):
        from model.check_uncertainty import true_worlds
        pred = ranked().sort_values("mix_id")
        truth = true_worlds(NM, SUN, 1000, 40, n=200, seed=99, error_scale=1, interaction=0)
        inside = (truth >= pred["efficiency_low"].values) & (truth <= pred["efficiency_high"].values)
        self.assertAlmostEqual(inside.mean(), 0.90, delta=0.04)


class WebApiTest(unittest.TestCase):
    def test_rejects_bad_input_without_network(self):
        for query in ("", "lat=abc&lon=1", "lat=51.5&lon=-0.1", "lat=3&lon=101&time=2026-11-20T05:00"):
            status, body = api_recommend(query)
            self.assertEqual(status, HTTPStatus.BAD_REQUEST, query)
            self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
