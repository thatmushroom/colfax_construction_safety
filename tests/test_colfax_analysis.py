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
