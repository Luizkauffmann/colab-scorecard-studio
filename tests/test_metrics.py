"""Issue #1: Gini must come from AUC (Gini = 2*AUC - 1), never 2*KS."""

import numpy as np
import pytest

from scorecard_studio import metrics as m


def test_perfect_and_random_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert m.auc_score(y, [1, 2, 3, 4, 5, 6]) == 1.0
    assert m.gini_score(y, [1, 2, 3, 4, 5, 6]) == 1.0
    assert m.auc_score(y, [1, 1, 1, 1, 1, 1]) == 0.5
    assert m.gini_score(y, [6, 5, 4, 3, 2, 1]) == -1.0


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.integers(0, 15, 300).astype(float)  # many ties
    pos, neg = s[y == 1], s[y == 0]
    pairwise = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert m.auc_score(y, s) == pytest.approx(pairwise, abs=1e-12)


def test_gini_is_not_two_times_ks():
    # Old engine returned gini = 2*ks. Construct a case where they clearly differ.
    events, non_events, woe = [10, 30, 60], [60, 30, 10], [-1.0, 0.0, 1.0]
    auc = m.auc_from_bins(events, non_events, woe)
    ks = m.ks_from_bins(events, non_events, woe)
    gini = m.gini_from_auc(auc)
    assert ks == pytest.approx(0.5)
    assert auc == pytest.approx((10 * (0 + 30) + 30 * (60 + 15) + 60 * (90 + 5)) / 10000)
    assert gini == pytest.approx(2 * auc - 1)
    assert gini != pytest.approx(2 * ks)


def test_binned_metrics_equal_row_level_metrics():
    rng = np.random.default_rng(1)
    bins = rng.integers(0, 5, 2000)
    y = (rng.random(2000) < 0.05 + 0.04 * bins).astype(int)
    woe_by_bin = np.array([-0.7, -0.2, 0.1, 0.4, 0.9])
    score = woe_by_bin[bins]
    ev = np.bincount(bins, weights=y, minlength=5)
    ne = np.bincount(bins, weights=1 - y, minlength=5)
    assert m.auc_from_bins(ev, ne, woe_by_bin) == pytest.approx(m.auc_score(y, score))
    assert m.ks_from_bins(ev, ne, woe_by_bin) == pytest.approx(m.ks_score(y, score))


def test_ks_uses_score_order_not_bin_order():
    # Non-monotonic bins: KS must be computed after sorting by WOE.
    events, non_events = [30, 5, 40], [20, 60, 20]
    woe = [np.log((e / 75) / (n / 100)) for e, n in zip(events, non_events)]
    in_bin_order = max(abs(np.cumsum(events) / 75 - np.cumsum(non_events) / 100))
    assert m.ks_from_bins(events, non_events, woe) > in_bin_order


def test_woe_iv_basic_and_zero_cells():
    w, iv, adj = m.woe_iv(20, 80, 100, 400)
    assert w == pytest.approx(np.log(0.2 / 0.2)) and iv == pytest.approx(0) and not adj
    w, iv, adj = m.woe_iv(0, 50, 100, 400)
    assert adj and np.isfinite(w) and w < 0
    assert m.woe_iv(0, 0, 100, 400) == (0.0, 0.0, False)


def test_psi():
    assert m.psi([10, 20, 70], [10, 20, 70]) == pytest.approx(0)
    assert m.psi([50, 50], [90, 10]) > 0.25


def test_interpret_iv():
    assert m.interpret_iv(0.01) == "Useless"
    assert m.interpret_iv(0.35) == "Strong"
    assert m.interpret_iv(0.9).startswith("Very strong")
