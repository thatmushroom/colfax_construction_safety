import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import colfax_analysis as ca  # noqa: E402


def test_explode_streets_normalises_names():
    df = pl.DataFrame(
        {"addr": ["1800 BLOCK E COLFAX AVE", "N BROADWAY ST / E 14TH AVE", "700 BLK WEST COLFAX AVE"]}
    )
    streets = ca.explode_streets(df).get_column("street").to_list()
    assert streets == ["COLFAX AVE", "BROADWAY ST", "14TH AVE", "COLFAX AVE"]


def test_fatal_reconciliation_lower_bound():
    df = pl.DataFrame(
        {
            "incident_id": ["a", "b", "c", "d"],
            "year": [2024, 2024, 2024, 2024],
            "n_fatal": [2, None, 0, 0],
            "fatal_offense": [True, True, False, False],
        }
    )
    out = ca.fatal_reconciliation(df, dashboard={2024: 5}, first_year=2024)
    row = out.row(0, named=True)
    assert row["fatal_incidents"] == 2
    assert row["sum_fatalities"] == 2
    assert row["lower_bound"] == 3  # max(2,1) + max(0,1)
    assert row["gap"] == 2


def test_find_csv_prefers_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("COLFAX_CSV", str(tmp_path / "x.csv"))
    assert ca.find_csv() == tmp_path / "x.csv"


def test_find_csv_picks_newest(monkeypatch, tmp_path):
    import os

    monkeypatch.delenv("COLFAX_CSV", raising=False)
    old, new = tmp_path / "b_old.csv", tmp_path / "a_new.csv"
    old.write_text("x")
    new.write_text("x")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    assert ca.find_csv(tmp_path) == new


def test_seasonal_window_excludes_after_sep_27():
    from datetime import datetime

    df = pl.DataFrame(
        {
            "dt": [datetime(2026, 9, 27, 23), datetime(2026, 9, 28), datetime(2025, 12, 1), datetime(2025, 1, 1)],
        }
    ).with_columns(
        (
            (pl.col("dt").dt.month() < 9) | ((pl.col("dt").dt.month() == 9) & (pl.col("dt").dt.day() <= 27))
        ).alias("in_window")
    )
    assert ca.seasonal_window(df).height == 2


def test_assign_zones_nearest_line_and_extent():
    geom = {"x_west": 0.0, "x_east": 10000.0, "lines": {"colfax": (0.0, 0.0), "alt_13th": (0.0, -1000.0), "alt_14th": (0.0, -600.0)}}
    df = pl.DataFrame({"x": [5000.0, 5000.0, 5000.0, 20000.0, None], "y": [100.0, -550.0, 5000.0, 0.0, 0.0]})
    z = ca.assign_zones(df, geom, buffer_ft=250).get_column("zone").to_list()
    assert z == ["colfax", "alt_14th", "rest", "rest", None]


def test_bernoulli_llr_only_flags_low_rate_circles():
    llr = ca.bernoulli_llr(c=[0, 20], n=[50, 50], C=100, N=1000)
    assert llr[0] > 0  # 0 of 50 vs expected 5 -> cold
    assert llr[1] == 0  # 20 of 50 is a hot spot -> ignored


def test_did_recovers_known_effect():
    rng = np.random.default_rng(7351)
    rows = []
    for year in range(2013, 2027):
        post = year >= ca.FIRST_POST_YEAR
        for g, base in (("colfax", 40.0), ("alt", 20.0), ("rest", 400.0)):
            mu = base * (0.5 if (g == "colfax" and post) else 1.0)
            rows.append({"group": g, "year": year, "n": int(rng.poisson(mu))})
    counts = pl.DataFrame(rows)
    tab, _ = ca.did_poisson(counts)
    rr = tab.filter(pl.col("term") == "colfax_post")["rate_ratio"][0]
    assert 0.35 < rr < 0.7


def test_exact_conditional_poisson_matches_binomial_tail():
    counts = pl.DataFrame(
        {
            "group": ["colfax", "colfax", "rest", "rest"],
            "year": [2024, 2025, 2024, 2025],
            "n": [10, 5, 100, 100],
        }
    )
    r = ca.exact_conditional_poisson(counts, "colfax")
    assert abs(r["citywide_rate_ratio"] - 1.0) < 1e-12
    assert abs(r["p0_expected_post_share"] - 0.5) < 1e-12
    # P(X <= 5) for X ~ Binomial(15, 0.5)
    assert abs(r["p_lower_tail"] - 0.15087890625) < 1e-9
    assert r["ci_low"] < r["relative_rate_ratio"] < r["ci_high"]


def test_combine_corridor_sums_colfax_and_alt():
    counts = pl.DataFrame(
        {
            "group": ["colfax", "alt", "rest"] * 2,
            "year": [2024] * 3 + [2025] * 3,
            "n": [10, 4, 100, 3, 6, 90],
        }
    )
    c = ca.combine_corridor(counts)
    corridor = c.filter(pl.col("group") == "corridor").sort("year")["n"].to_list()
    assert corridor == [14, 9]
    assert c.filter(pl.col("group") == "rest").height == 2


def test_wide_groups_labels_streets_and_keeps_missing_coordinates():
    df = pl.DataFrame({"zone": ["colfax", "alt_13th", "alt_14th", "alt_16th", "alt_17th", "rest", None]})
    g = ca.wide_groups(df).get_column("group").to_list()
    assert g == ["colfax", "alt", "alt", "alt", "alt", "rest", None]
