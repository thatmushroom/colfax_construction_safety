"""Did Colfax construction reduce deaths / serious bodily injuries (SBIs)
beyond the citywide trend?

All data handling is done with polars. numpy / scipy / statsmodels are used
for the spatial and inferential steps.

Design
------
* Unit of analysis: crashes (one row per crash record).
* Outcome ("severe"): crashes whose top traffic-accident offense is
  ``TRAF - ACCIDENT - FATAL`` or ``TRAF - ACCIDENT - SBI``. This definition
  reproduces the 318 SBI crashes quoted for 2025. A secondary definition uses
  the SERIOUSLY_INJURED / FATALITIES count columns.
* Seasonality control: only Jan 1 - Sep 27 of every year is used, and nothing
  after 2026-09-27.
* Pre period: 2013-2024 (Jan-Sep 27). Construction period: 2025-2026
  (Jan-Sep 27). Oct-Dec 2024 is construction but is excluded by the
  seasonality rule.
* Zones (distance to the Colfax / 13th / 14th Ave centerlines, in feet, in
  Colorado State Plane Central, EPSG:2232): ``colfax``, ``alt_13th``,
  ``alt_14th`` between the Broadway and Colorado Blvd intersections, and
  ``rest`` (everything else = citywide baseline).
"""

from __future__ import annotations

import math
import os
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import sparse, stats
from scipy.spatial import cKDTree
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "analysis" / "output"
FIG_DIR = ROOT / "analysis" / "figures"

CUTOFF_MONTH, CUTOFF_DAY = 9, 27
# Full-calendar version (seasonality handled in the model, no Jan-Sep window)
CONSTRUCTION_START = date(2024, 10, 1)  # first month under construction
FULLCAL_END_EXCL = datetime(2026, 9, 28)  # construction ended Sep 27, 2026
MONTHLY_KEYS = ("ym", "month", "days")  # extra columns carried by monthly counts
FIRST_POST_YEAR = 2025  # Jan-Sep of 2025 and 2026 are under construction
SEVERE_OFFENSES = ("TRAF - ACCIDENT - FATAL", "TRAF - ACCIDENT - SBI")
ZONE_ORDER = ["colfax", "alt_13th", "alt_14th", "rest"]


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
def find_csv(data_dir: Path = DATA_DIR) -> Path:
    """Return the CSV to analyse: $COLFAX_CSV if set, else the most recently
    modified ``*.csv`` in ``data_dir`` (so a fresh download wins over older ones)."""
    override = os.environ.get("COLFAX_CSV")
    if override:
        return Path(override)
    files = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No CSV found in {data_dir}")
    return files[-1]


def load_crashes(path: Path | None = None) -> pl.DataFrame:
    """Read the raw crash CSV and return a tidy crash-level frame."""
    path = path or find_csv()
    raw = pl.read_csv(path, infer_schema_length=0)
    return raw.select(
        pl.col("incident_id"),
        pl.col("first_occurrence_date")
        .str.to_datetime("%m/%d/%Y %I:%M:%S %p", strict=False)
        .alias("dt"),
        pl.col("incident_address").str.strip_chars().str.to_uppercase().alias("addr"),
        pl.col("geo_lon").cast(pl.Float64, strict=False).alias("lon"),
        pl.col("geo_lat").cast(pl.Float64, strict=False).alias("lat"),
        pl.col("geo_x").cast(pl.Float64, strict=False).alias("x"),
        pl.col("geo_y").cast(pl.Float64, strict=False).alias("y"),
        pl.col("SERIOUSLY_INJURED").cast(pl.Int64, strict=False).alias("n_sbi"),
        pl.col("FATALITIES").cast(pl.Int64, strict=False).alias("n_fatal"),
        pl.col("top_traffic_accident_offense").str.strip_chars().alias("offense"),
    ).with_columns(
        pl.col("dt").dt.year().alias("year"),
        pl.col("offense").is_in(SEVERE_OFFENSES).alias("severe"),
        (
            (pl.col("n_sbi").fill_null(0) > 0) | (pl.col("n_fatal").fill_null(0) > 0)
        ).alias("severe_counts"),
        (pl.col("offense") == "TRAF - ACCIDENT - FATAL").alias("fatal_offense"),
        (
            (pl.col("dt").dt.month() < CUTOFF_MONTH)
            | (
                (pl.col("dt").dt.month() == CUTOFF_MONTH)
                & (pl.col("dt").dt.day() <= CUTOFF_DAY)
            )
        ).alias("in_window"),
    )


def seasonal_window(df: pl.DataFrame) -> pl.DataFrame:
    """Keep Jan 1 - Sep 27 of each year (also enforces the 2026-09-27 cut-off)."""
    return df.filter(pl.col("in_window"))


def add_period(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("year") >= FIRST_POST_YEAR)
        .then(pl.lit("construction"))
        .otherwise(pl.lit("pre"))
        .alias("period")
    )


def reconcile_2025(df: pl.DataFrame) -> pl.DataFrame:
    """Full-year 2025 totals under alternative definitions, for comparison with
    the user-quoted 94 deaths / 318 SBI events."""
    y = df.filter(pl.col("year") == 2025)
    return pl.DataFrame(
        {
            "measure": [
                "SBI-offense crashes",
                "FATAL-offense crashes",
                "crashes with FATALITIES > 0",
                "persons killed (sum FATALITIES)",
                "crashes with SERIOUSLY_INJURED > 0",
                "persons seriously injured (sum SERIOUSLY_INJURED)",
            ],
            "value_2025_full_year": [
                y.filter(pl.col("offense") == "TRAF - ACCIDENT - SBI").height,
                y.filter(pl.col("fatal_offense")).height,
                y.filter(pl.col("n_fatal").fill_null(0) > 0).height,
                int(y.get_column("n_fatal").sum()),
                y.filter(pl.col("n_sbi").fill_null(0) > 0).height,
                int(y.get_column("n_sbi").sum()),
            ],
        }
    )


def fatal_reconciliation(
    df: pl.DataFrame, dashboard: dict[int, int] | None = None, first_year: int = 2022
) -> pl.DataFrame:
    """Per-year deaths implied by the CSV, set against dashboard totals.

    ``FATALITIES`` is blank or 0 for many FATAL-coded crashes, so summing it
    undercounts. The lower bound counts every incident that is FATAL-coded or
    has ``FATALITIES`` > 0 as at least one death: ``max(FATALITIES, 1)``.
    """
    out = (
        df.filter(pl.col("fatal_offense") | (pl.col("n_fatal").fill_null(0) > 0))
        .group_by("incident_id")
        .agg(
            pl.col("year").min(),
            pl.col("n_fatal").max().fill_null(0).alias("recorded"),
        )
        .with_columns(pl.max_horizontal("recorded", pl.lit(1)).alias("lower_bound"))
        .group_by("year")
        .agg(
            pl.len().alias("fatal_incidents"),
            pl.col("recorded").sum().alias("sum_fatalities"),
            pl.col("lower_bound").sum().alias("lower_bound"),
        )
        .filter(pl.col("year") >= first_year)
        .sort("year")
    )
    if dashboard:
        out = out.with_columns(
            pl.col("year").replace_strict(dashboard, default=None).alias("dashboard")
        ).with_columns((pl.col("dashboard") - pl.col("lower_bound")).alias("gap"))
    return out


def offense_vs_counts(df: pl.DataFrame, first_year: int = 2018) -> pl.DataFrame:
    """Per year: how often the SBI / FATAL offense codes agree with the
    SERIOUSLY_INJURED / FATALITIES count columns (full years, all dates)."""
    return (
        df.filter(pl.col("year") >= first_year)
        .group_by("year")
        .agg(
            (pl.col("offense") == "TRAF - ACCIDENT - SBI").sum().alias("sbi_offense"),
            ((pl.col("offense") == "TRAF - ACCIDENT - SBI") & (pl.col("n_sbi") == 0)).sum().alias("sbi_offense_count_zero"),
            pl.col("fatal_offense").sum().alias("fatal_offense"),
            (pl.col("fatal_offense") & (pl.col("n_fatal") == 0)).sum().alias("fatal_offense_count_zero"),
            (pl.col("n_sbi").fill_null(0) > 0).sum().alias("crashes_sbi_count"),
            (pl.col("n_fatal").fill_null(0) > 0).sum().alias("crashes_fatal_count"),
            pl.col("n_fatal").sum().alias("people_killed"),
            pl.col("n_sbi").is_null().mean().round(3).alias("share_counts_null"),
        )
        .with_columns(
            (pl.col("sbi_offense_count_zero") / pl.col("sbi_offense")).round(2).alias("sbi_share_zero"),
            (pl.col("fatal_offense_count_zero") / pl.col("fatal_offense")).round(2).alias("fatal_share_zero"),
        )
        .sort("year")
    )


def definitions_by_year(df: pl.DataFrame, first_year: int = 2023) -> pl.DataFrame:
    """Jan 1 - Sep 27 severe-crash counts under both definitions, with
    citywide (all-zone) totals, by year."""
    return (
        seasonal_window(df)
        .filter(pl.col("year") >= first_year)
        .group_by("year")
        .agg(
            pl.col("severe").sum().alias("A_offense_code"),
            pl.col("severe_counts").sum().alias("B_count_columns"),
            pl.col("fatal_offense").sum().alias("deaths_A_fatal_offense"),
            (pl.col("n_fatal").fill_null(0) > 0).sum().alias("deaths_B_fatality_count"),
        )
        .sort("year")
    )


def missingness(df: pl.DataFrame, outcome: str = "severe") -> pl.DataFrame:
    """Share of outcome crashes lacking coordinates, by year (in window)."""
    return (
        seasonal_window(df)
        .filter(pl.col(outcome))
        .group_by("year")
        .agg(
            pl.len().alias("n_severe"),
            pl.col("x").is_null().sum().alias("n_no_coords"),
        )
        .with_columns((pl.col("n_no_coords") / pl.col("n_severe")).alias("share_no_coords"))
        .sort("year")
    )


# --------------------------------------------------------------------------
# Corridor geometry and zone assignment
# --------------------------------------------------------------------------
STREET_PATTERNS = {
    "colfax": r"\bCOLFAX AVE\b",
    "alt_13th": r"\b13TH AVE\b",
    "alt_14th": r"\b14TH AVE\b",
}


# Wider corridor: adds 16th and 17th Ave (north of Colfax) to Colfax, 13th and 14th Ave
WIDE_PATTERNS = {
    **STREET_PATTERNS,
    "alt_16th": r"\b16TH AVE\b",
    "alt_17th": r"\b17TH AVE\b",
}


def corridor_geometry(df: pl.DataFrame, patterns: dict | None = None) -> dict:
    """Estimate the study extent and street centerlines from the data.

    Extent: median x of crashes recorded at the Colfax/Broadway and
    Colfax/Colorado Blvd intersections. Centerlines: straight line fit
    (y ~ x) through address-matched crashes within the extent.
    """
    a = df.filter(pl.col("x").is_not_null())
    colfax_addr = pl.col("addr").str.contains(STREET_PATTERNS["colfax"])
    x_w = a.filter(colfax_addr & pl.col("addr").str.contains("BROADWAY")).get_column("x").median()
    x_e = a.filter(colfax_addr & pl.col("addr").str.contains("COLORADO BLVD")).get_column("x").median()
    lines = {}
    for name, pat in (patterns or STREET_PATTERNS).items():
        s = a.filter(
            pl.col("addr").str.contains(pat)
            & pl.col("x").is_between(x_w, x_e)
        )
        slope, intercept = np.polyfit(s.get_column("x").to_numpy(), s.get_column("y").to_numpy(), 1)
        lines[name] = (float(slope), float(intercept))
    return {"x_west": float(x_w), "x_east": float(x_e), "lines": lines}


def _dist_to_line(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return np.abs(slope * x - y + intercept) / np.hypot(slope, 1.0)


def assign_zones(
    df: pl.DataFrame,
    geom: dict,
    buffer_ft: float = 250.0,
    end_tol_ft: float = 100.0,
) -> pl.DataFrame:
    """Add a ``zone`` column. Crashes without coordinates get ``None`` (they
    are dropped from spatial analyses and reported in :func:`missingness`)."""
    x = df.get_column("x").to_numpy()
    y = df.get_column("y").to_numpy()
    valid = ~np.isnan(x) & ~np.isnan(y)
    in_extent = valid & (x >= geom["x_west"] - end_tol_ft) & (x <= geom["x_east"] + end_tol_ft)
    names = list(geom["lines"])
    d = np.full((len(names), len(x)), np.inf)
    for i, n in enumerate(names):
        s, b = geom["lines"][n]
        d[i, valid] = _dist_to_line(x[valid], y[valid], s, b)
    nearest = d.argmin(axis=0)
    within = d.min(axis=0) <= buffer_ft
    zone = np.where(valid, "rest", None).astype(object)
    hit = in_extent & within
    zone[hit] = np.array(names, dtype=object)[nearest[hit]]
    return df.with_columns(pl.Series("zone", zone, dtype=pl.String))


def zone_group(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse alt_13th / alt_14th into a single ``alt`` group."""
    return df.with_columns(
        pl.when(pl.col("zone").is_in(["alt_13th", "alt_14th"]))
        .then(pl.lit("alt"))
        .otherwise(pl.col("zone"))
        .alias("group")
    )


# --------------------------------------------------------------------------
# Counts and the difference-in-differences models
# --------------------------------------------------------------------------
def yearly_counts(df: pl.DataFrame, outcome: str = "severe", by: str = "group") -> pl.DataFrame:
    """Zone x year counts of outcome crashes (complete grid, zeros filled)."""
    d = seasonal_window(df).filter(pl.col(by).is_not_null())
    years = sorted(d.get_column("year").unique().to_list())
    groups = sorted(d.get_column(by).unique().to_list())
    grid = pl.DataFrame({by: groups}).join(pl.DataFrame({"year": years}), how="cross")
    n = d.filter(pl.col(outcome)).group_by(by, "year").agg(pl.len().alias("n"))
    return (
        grid.join(n, on=[by, "year"], how="left")
        .with_columns(pl.col("n").fill_null(0))
        .sort(by, "year")
    )


def _model_frame(counts: pl.DataFrame, drop_years=(), pre_years=None) -> "pd.DataFrame":  # noqa: F821
    c = counts.filter(~pl.col("year").is_in(list(drop_years)))
    if pre_years is not None:
        keep = set(pre_years) | {y for y in c.get_column("year").to_list() if y >= FIRST_POST_YEAR}
        c = c.filter(pl.col("year").is_in(list(keep)))
    c = c.with_columns(
        (pl.col("year") >= FIRST_POST_YEAR).cast(pl.Int8).alias("post"),
        (pl.col("group") == "colfax").cast(pl.Int8).alias("is_colfax"),
        (pl.col("group") == "alt").cast(pl.Int8).alias("is_alt"),
    ).with_columns(
        (pl.col("is_colfax") * pl.col("post")).alias("colfax_post"),
        (pl.col("is_alt") * pl.col("post")).alias("alt_post"),
        (pl.col("is_colfax") * (pl.col("year") == 2025)).alias("colfax_2025"),
        (pl.col("is_colfax") * (pl.col("year") == 2026)).alias("colfax_2026"),
        (pl.col("is_alt") * (pl.col("year") == 2025)).alias("alt_2025"),
        (pl.col("is_alt") * (pl.col("year") == 2026)).alias("alt_2026"),
        (pl.col("year") - 2018.5).alias("yc"),
    )
    return c.to_pandas()


def _fit_glm(formula: str, data, offset=None):
    """Poisson GLM; SEs are inflated by the Pearson dispersion when it exceeds 1
    (quasi-Poisson) but never deflated (these counts are small, so we stay
    conservative when the data look under-dispersed). ``offset`` is an optional
    log-exposure vector."""
    kw = {} if offset is None else {"offset": np.asarray(offset, dtype=float)}
    fit = smf.glm(formula, data, family=sm.families.Poisson(), **kw).fit()
    dispersion = float(fit.pearson_chi2 / fit.df_resid)
    qfit = smf.glm(formula, data, family=sm.families.Poisson(), **kw).fit(
        scale=dispersion if dispersion > 1 else 1.0
    )
    return fit, qfit, dispersion


def _tidy(fit, scale_fit, terms: list[str]) -> pl.DataFrame:
    ci = scale_fit.conf_int()
    rows = []
    for t in terms:
        rows.append(
            {
                "term": t,
                "rate_ratio": float(np.exp(scale_fit.params[t])),
                "ci_low": float(np.exp(ci.loc[t, 0])),
                "ci_high": float(np.exp(ci.loc[t, 1])),
                "p_value": float(scale_fit.pvalues[t]),
                "p_poisson": float(fit.pvalues[t]),
            }
        )
    return pl.DataFrame(rows)


def did_poisson(
    counts: pl.DataFrame,
    drop_years=(),
    pre_years=None,
    by_year: bool = False,
) -> tuple[pl.DataFrame, float]:
    """Difference-in-differences Poisson GLM on zone x year counts.

    ``n ~ C(year) + is_colfax + is_alt + colfax_post + alt_post``. The year
    fixed effects absorb the citywide (``rest``) trend, so ``colfax_post`` is
    the Colfax rate ratio (construction / pre) *relative to* the citywide
    rate ratio. Standard errors are quasi-Poisson (Pearson chi-square scale).
    Returns (tidy table, Pearson dispersion).
    """
    df = _model_frame(counts, drop_years, pre_years)
    if by_year:
        f = "n ~ C(year) + is_colfax + is_alt + colfax_2025 + colfax_2026 + alt_2025 + alt_2026"
        terms = ["colfax_2025", "colfax_2026", "alt_2025", "alt_2026"]
    else:
        f = "n ~ C(year) + is_colfax + is_alt + colfax_post + alt_post"
        terms = ["colfax_post", "alt_post"]
    fit, qfit, dispersion = _fit_glm(f, df)
    return _tidy(fit, qfit, terms), dispersion


def did_poisson_trend(counts: pl.DataFrame, drop_years=()) -> pl.DataFrame:
    """DiD with zone-specific linear time trends. The trends are estimated
    mostly from the pre years, so this relaxes the parallel-trends assumption
    (Colfax's crash counts were rising before construction)."""
    df = _model_frame(counts, drop_years)
    f = (
        "n ~ C(year) + is_colfax + is_alt + is_colfax:yc + is_alt:yc"
        " + colfax_post + alt_post"
    )
    fit, qfit, _ = _fit_glm(f, df)
    return _tidy(fit, qfit, ["colfax_post", "alt_post", "is_colfax:yc", "is_alt:yc"])


def did_negbin(counts: pl.DataFrame, drop_years=()) -> tuple[pl.DataFrame, float]:
    """Same design as :func:`did_poisson` but with a negative-binomial (NB2)
    likelihood whose dispersion ``alpha`` is estimated by maximum likelihood.

    If the counts are not over-dispersed, alpha is estimated at (or near) the
    lower boundary of 0 and the model collapses to Poisson; the returned alpha
    shows when that has happened. Returns (tidy table, alpha).
    """
    df = _model_frame(counts, drop_years)
    f = "n ~ C(year) + is_colfax + is_alt + colfax_post + alt_post"
    with warnings.catch_warnings():
        # alpha sits on its boundary when the data are not over-dispersed, which
        # makes the optimiser report non-convergence; that case is flagged via alpha
        warnings.simplefilter("ignore")
        fit = smf.negativebinomial(f, df).fit(disp=0, maxiter=500, method="bfgs")
    ci = fit.conf_int()
    rows = [
        {
            "term": t,
            "rate_ratio": float(np.exp(fit.params[t])),
            "ci_low": float(np.exp(ci.loc[t, 0])),
            "ci_high": float(np.exp(ci.loc[t, 1])),
            "p_value": float(fit.pvalues[t]),
        }
        for t in ("colfax_post", "alt_post")
    ]
    return pl.DataFrame(rows), float(fit.params["alpha"])


def exact_conditional_poisson(
    counts: pl.DataFrame, group: str, drop_years=(), pre_years=None
) -> dict:
    """Exact conditional test of one zone's change against the citywide change.

    Let theta be the rest-of-city rate ratio (construction rate / pre rate,
    per year). Under the null that the zone changed like the rest of the city,
    its construction-period count, conditional on its total count n, is
    Binomial(n, p0) with p0 = theta * Y_post / (Y_pre + theta * Y_post), where
    Y are the numbers of years. This needs no large-sample approximation, so
    it suits a handful of events. The CI is Clopper-Pearson on the binomial
    proportion, converted to a ratio of rate ratios (zone / citywide).

    ``theta`` is treated as known (the rest-of-city counts are in the
    hundreds, so its sampling error is small by comparison).
    """
    c = counts.filter(~pl.col("year").is_in(list(drop_years)))
    if pre_years is not None:
        keep = set(pre_years) | {y for y in c.get_column("year").to_list() if y >= FIRST_POST_YEAR}
        c = c.filter(pl.col("year").is_in(list(keep)))
    c = c.with_columns((pl.col("year") >= FIRST_POST_YEAR).alias("post"))

    def tot(g, post):
        return int(c.filter((pl.col("group") == g) & (pl.col("post") == post))["n"].sum())

    y_pre = c.filter(~pl.col("post")).get_column("year").n_unique()
    y_post = c.filter(pl.col("post")).get_column("year").n_unique()
    a, b = tot(group, False), tot(group, True)
    theta = (tot("rest", True) / y_post) / (tot("rest", False) / y_pre)
    n = a + b
    p0 = theta * y_post / (y_pre + theta * y_post)
    p_low = float(stats.binomtest(b, n, p0, alternative="less").pvalue)
    p_two = float(stats.binomtest(b, n, p0, alternative="two-sided").pvalue)
    ci = stats.binomtest(b, n, p0).proportion_ci(confidence_level=0.95, method="exact")

    def to_ratio(p):  # ratio of rate ratios, zone vs citywide
        return (p / (1 - p)) * (y_pre / y_post) / theta if 0 < p < 1 else (0.0 if p == 0 else np.inf)

    return {
        "group": group,
        "pre_count": a,
        "post_count": b,
        "years_pre": y_pre,
        "years_post": y_post,
        "citywide_rate_ratio": float(theta),
        "p0_expected_post_share": float(p0),
        "expected_post_count": float(n * p0),
        "relative_rate_ratio": float(to_ratio(b / n)),
        "ci_low": float(to_ratio(ci.low)),
        "ci_high": float(to_ratio(ci.high)),
        "p_lower_tail": p_low,
        "p_two_sided": p_two,
    }


def combine_corridor(counts: pl.DataFrame) -> pl.DataFrame:
    """Merge the Colfax and 13th/14th Ave groups into one ``corridor`` unit
    (zone x year counts), leaving ``rest`` unchanged."""
    return (
        counts.with_columns(
            pl.when(pl.col("group").is_in(["colfax", "alt"]))
            .then(pl.lit("corridor"))
            .otherwise(pl.col("group"))
            .alias("group")
        )
        .group_by("group", "year", *[c for c in MONTHLY_KEYS if c in counts.columns])
        .agg(pl.col("n").sum())
        .sort("group", "year", *[c for c in MONTHLY_KEYS if c in counts.columns])
    )


def did_combined(
    counts: pl.DataFrame, drop_years=(), pre_years=None, by_year: bool = False, trend: bool = False
) -> pl.DataFrame:
    """Poisson difference-in-differences for the combined corridor (Colfax +
    13th/14th Ave) versus the rest of the city. ``counts`` must come from
    :func:`combine_corridor`. Year fixed effects absorb the citywide trend, so
    ``corridor_post`` is the corridor's rate ratio relative to the citywide one.
    """
    c = counts.filter(~pl.col("year").is_in(list(drop_years)))
    if pre_years is not None:
        keep = set(pre_years) | {y for y in c.get_column("year").to_list() if y >= FIRST_POST_YEAR}
        c = c.filter(pl.col("year").is_in(list(keep)))
    d = c.with_columns(
        (pl.col("group") == "corridor").cast(pl.Int8).alias("is_corridor"),
        (pl.col("year") >= FIRST_POST_YEAR).cast(pl.Int8).alias("post"),
        (pl.col("year") - 2018.5).alias("yc"),
    ).with_columns(
        (pl.col("is_corridor") * pl.col("post")).alias("corridor_post"),
        (pl.col("is_corridor") * (pl.col("year") == 2025)).alias("corridor_2025"),
        (pl.col("is_corridor") * (pl.col("year") == 2026)).alias("corridor_2026"),
    ).to_pandas()
    if by_year:
        f, terms = "n ~ C(year) + is_corridor + corridor_2025 + corridor_2026", ["corridor_2025", "corridor_2026"]
    elif trend:
        f, terms = "n ~ C(year) + is_corridor + is_corridor:yc + corridor_post", ["corridor_post", "is_corridor:yc"]
    else:
        f, terms = "n ~ C(year) + is_corridor + corridor_post", ["corridor_post"]
    fit, qfit, _ = _fit_glm(f, d)
    return _tidy(fit, qfit, terms)


def combined_analysis(df: pl.DataFrame, outcome: str = "severe") -> dict:
    """All tests for the combined corridor unit vs the rest of the city."""
    base = yearly_counts(df, outcome)
    comb = combine_corridor(base)
    out = {"outcome": outcome, "counts": comb}
    out["did"] = did_combined(comb)
    out["did_by_year"] = did_combined(comb, by_year=True)
    out["did_trend"] = did_combined(comb, trend=True)
    out["did_no2020"] = did_combined(comb, drop_years=(2020,))
    out["did_2021_on"] = did_combined(comb, pre_years=range(2021, 2025))
    out["exact"] = pl.DataFrame(
        [
            exact_conditional_poisson(comb, "corridor") | {"pre_period": "2013-2024"},
            exact_conditional_poisson(comb, "corridor", pre_years=range(2021, 2025)) | {"pre_period": "2021-2024"},
        ]
    )
    # the same split but placing Colfax and alt routes as separate rows, for reference
    out["components"] = (
        base.filter(pl.col("group") != "rest")
        .with_columns((pl.col("year") >= FIRST_POST_YEAR).alias("construction"))
        .group_by("group", "construction")
        .agg(pl.col("n").sum().alias("n"), pl.col("year").n_unique().alias("years"))
        .with_columns((pl.col("n") / pl.col("years")).round(2).alias("per_year"))
        .sort("group", "construction")
    )
    return out


def leave_one_year_out(df: pl.DataFrame, outcome: str = "severe") -> pl.DataFrame:
    """Re-estimate the difference-in-differences leaving out each pre-period
    year in turn. Returns one row per (omitted year, unit) with the rate ratio
    relative to the citywide change. Units: ``colfax``, ``alt`` (13th/14th Ave,
    from the three-group model) and ``corridor`` (Colfax + 13th/14th combined).
    """
    base = yearly_counts(df, outcome)
    comb = combine_corridor(base)
    pre_years = sorted(y for y in base.get_column("year").unique().to_list() if y < FIRST_POST_YEAR)
    rows = []
    for y in [None] + pre_years:
        drop = () if y is None else (y,)
        label = "none (all years)" if y is None else str(y)
        tab, _ = did_poisson(base, drop_years=drop)
        for r in tab.iter_rows(named=True):
            rows.append({"omitted_year": label, "unit": r["term"].replace("_post", "")} | {k: r[k] for k in ("rate_ratio", "ci_low", "ci_high", "p_value")})
        for r in did_combined(comb, drop_years=drop).iter_rows(named=True):
            rows.append({"omitted_year": label, "unit": "corridor"} | {k: r[k] for k in ("rate_ratio", "ci_low", "ci_high", "p_value")})
    return pl.DataFrame(rows)


def prepare_wide(buffer_ft: float = 250.0) -> tuple[pl.DataFrame, dict]:
    """Like :func:`prepare` but for the wide corridor: Colfax, 13th, 14th, 16th
    and 17th Ave between Broadway and Colorado Blvd. ``group`` is ``colfax``,
    ``alt`` (the four other streets) or ``rest`` (everything else), so the
    existing combined / leave-one-out functions apply unchanged and merge the
    five streets into one ``corridor`` unit.
    """
    df = add_period(load_crashes())
    geom = corridor_geometry(df, WIDE_PATTERNS)
    return wide_groups(assign_zones(df, geom, buffer_ft=buffer_ft)), geom


def wide_groups(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("zone").is_null())
        .then(None)
        .when(pl.col("zone") == "colfax")
        .then(pl.lit("colfax"))
        .when(pl.col("zone") == "rest")
        .then(pl.lit("rest"))
        .otherwise(pl.lit("alt"))
        .alias("group")
    )


def wide_analysis(df_w: pl.DataFrame, outcome: str = "severe") -> dict:
    """Tests for the wide corridor (five streets merged) against the rest of
    the city, plus a per-street breakdown."""
    out = combined_analysis(df_w, outcome)
    base = yearly_counts(df_w, outcome)
    out["did_three_group"], _ = did_poisson(base)  # Colfax vs the four other streets vs rest
    out["fisher"] = exact_share_test(out["counts"], "corridor")
    zc = yearly_counts(df_w, outcome, by="zone")
    out["by_street"] = (
        zc.filter(pl.col("zone") != "rest")
        .with_columns((pl.col("year") >= FIRST_POST_YEAR).alias("construction"))
        .group_by("zone", "construction")
        .agg(pl.col("n").sum().alias("n"), pl.col("year").n_unique().alias("years"))
        .with_columns((pl.col("n") / pl.col("years")).round(2).alias("per_year"))
        .sort("zone", "construction")
    )
    out["street_counts"] = zc.filter(pl.col("zone") != "rest")
    out["loo"] = leave_one_year_out(df_w, outcome)
    try:
        nb, alpha = did_negbin(base)
        out["nb_alpha"] = alpha
    except Exception:
        out["nb_alpha"] = float("nan")
    return out


def wide_buffer_sensitivity(
    df_w: pl.DataFrame, geom: dict, buffers=(100.0, 150.0, 250.0, 280.0), outcome: str = "severe"
) -> pl.DataFrame:
    """Wide-corridor rate ratio for several buffer widths (feet). The five
    centerlines are about 570-590 ft apart, so buffers above ~285 ft would
    overlap."""
    rows = []
    for b in buffers:
        d = wide_groups(assign_zones(df_w.drop("zone", "group", strict=False), geom, buffer_ft=b))
        counts = combine_corridor(yearly_counts(d, outcome))
        r = did_combined(counts).row(0, named=True)
        rows.append(
            {"buffer_ft": b, "n_corridor": int(counts.filter(pl.col("group") == "corridor")["n"].sum())}
            | {k: r[k] for k in ("rate_ratio", "ci_low", "ci_high", "p_value")}
        )
    return pl.DataFrame(rows)


def pretrend_test(counts: pl.DataFrame, drop_years=()) -> pl.DataFrame:
    """Do Colfax / alt routes already trend differently from the rest of the
    city before construction? (pre years only; zone-specific linear trends)."""
    df = _model_frame(counts, drop_years)
    df = df[df["post"] == 0]
    f = "n ~ C(year) + is_colfax + is_alt + is_colfax:yc + is_alt:yc"
    fit, qfit, _ = _fit_glm(f, df)
    return _tidy(fit, qfit, ["is_colfax:yc", "is_alt:yc"])


def exact_share_test(counts: pl.DataFrame, group: str, drop_years=()) -> dict:
    """Fisher exact test on the 2x2 table (group vs rest) x (pre vs construction).

    Ignores year-to-year structure, so it is a coarse, assumption-light
    complement to the GLM.
    """
    c = counts.filter(~pl.col("year").is_in(list(drop_years))).with_columns(
        (pl.col("year") >= FIRST_POST_YEAR).alias("post")
    )
    g = c.filter(pl.col("group") == group).group_by("post").agg(pl.col("n").sum())
    r = c.filter(pl.col("group") == "rest").group_by("post").agg(pl.col("n").sum())
    gp = dict(zip(g["post"].to_list(), g["n"].to_list()))
    rp = dict(zip(r["post"].to_list(), r["n"].to_list()))
    table = [[gp[False], gp[True]], [rp[False], rp[True]]]
    odds, p = stats.fisher_exact(table)
    ci = stats.contingency.odds_ratio(table).confidence_interval()
    return {
        "group": group,
        "pre_group": table[0][0],
        "post_group": table[0][1],
        "pre_rest": table[1][0],
        "post_rest": table[1][1],
        "odds_ratio": float(odds),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "p_value": float(p),
    }


# --------------------------------------------------------------------------
# Placebo-corridor permutation test
# --------------------------------------------------------------------------
def explode_streets(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (crash, street) with a normalised street key."""
    return (
        df.filter(pl.col("addr").is_not_null())
        .with_columns(pl.col("addr").str.split(" / ").alias("street"))
        .explode("street")
        .with_columns(
            pl.col("street")
            .str.replace(r"^\d+\s+((BLOCK|BLK)\s+)?", "")
            .str.replace(r"^(EAST|WEST|NORTH|SOUTH|E|W|N|S)\s+", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
    )


def placebo_windows(
    df: pl.DataFrame,
    window_ft: float,
    step_ft: float = 1320.0,
    min_street_crashes: int = 500,
    exclude_streets=("COLFAX AVE", "13TH AVE", "14TH AVE"),
    outcome: str = "severe",
) -> pl.DataFrame:
    """Slide a window of the same length as the Colfax study segment along
    every long street and tally severe crashes pre vs construction.

    Street position is the projection on the street's principal axis, so
    mildly curved streets are handled approximately.
    """
    d = seasonal_window(df).filter(pl.col("x").is_not_null())
    s = explode_streets(d).filter(~pl.col("street").is_in(list(exclude_streets)))
    keep = (
        s.group_by("street")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_street_crashes)
        .get_column("street")
        .to_list()
    )
    rows = []
    for street in keep:
        ss = s.filter(pl.col("street") == street)
        xy = ss.select("x", "y").to_numpy()
        xy_c = xy - xy.mean(axis=0)
        axis = np.linalg.svd(xy_c, full_matrices=False)[2][0]
        t = xy_c @ axis
        lo, hi = np.percentile(t, [1, 99])
        if hi - lo < window_ft:
            continue
        sev = ss.get_column(outcome).to_numpy()
        post = (ss.get_column("year").to_numpy() >= FIRST_POST_YEAR)
        starts = np.arange(lo, hi - window_ft + 1, step_ft)
        for k, a in enumerate(starts):
            inside = (t >= a) & (t <= a + window_ft)
            rows.append(
                {
                    "street": street,
                    "start_ft": float(a),
                    "k": k,
                    "n_pre": int((inside & sev & ~post).sum()),
                    "n_post": int((inside & sev & post).sum()),
                }
            )
    return pl.DataFrame(rows)


def log_rate_ratio(n_pre, n_post, years_pre: int, years_post: int):
    """Log of (post rate / pre rate) per year, with a 0.5 continuity correction."""
    return np.log((np.asarray(n_post) + 0.5) / years_post) - np.log(
        (np.asarray(n_pre) + 0.5) / years_pre
    )


def placebo_test(
    windows: pl.DataFrame,
    colfax_pre: int,
    colfax_post: int,
    years_pre: int,
    years_post: int,
    baseline_ratio: float = 0.5,
    stride: int = 3,
) -> dict:
    """Empirical (one-sided, lower-tail) p-value for Colfax's change relative
    to placebo windows of the same length that had a comparable baseline
    (``n_pre >= baseline_ratio * colfax_pre``). Conditioning on baseline
    guards against regression to the mean being mistaken for a treatment
    effect. Windows overlap, so an independent (non-overlapping) subset is
    also reported.
    """
    w = windows.with_columns(
        pl.Series("lrr", log_rate_ratio(windows["n_pre"], windows["n_post"], years_pre, years_post))
    )
    obs = float(log_rate_ratio(colfax_pre, colfax_post, years_pre, years_post))
    elig = w.filter(pl.col("n_pre") >= baseline_ratio * colfax_pre)
    # non-overlapping subset: every ``stride``-th start (= window_ft / step_ft) per street
    indep = elig.filter(pl.col("k") % stride == 0)

    def p_lower(x: pl.DataFrame) -> float:
        n = x.height
        return float((1 + (x["lrr"] <= obs).sum()) / (1 + n))

    return {
        "observed_lrr": obs,
        "observed_ratio": float(np.exp(obs)),
        "n_windows": elig.height,
        "p_all_windows": p_lower(elig),
        "n_independent": indep.height,
        "p_independent": p_lower(indep),
        "percentile": float((elig["lrr"] <= obs).mean()),
        "placebo": elig,
    }


# --------------------------------------------------------------------------
# Spatial scan statistic (Bernoulli, cold clusters of construction-period crashes)
# --------------------------------------------------------------------------
def bernoulli_llr(c, n, C, N):
    """Kulldorff Bernoulli log-likelihood ratio for a circle with ``c`` cases of
    ``n`` points, versus ``C`` of ``N`` overall. Only *low*-rate circles
    (c / n < C / N) get a positive value."""
    c = np.asarray(c, dtype=float)
    n = np.asarray(n, dtype=float)
    inside = xlogy(c, c / np.maximum(n, 1)) + xlogy(n - c, (n - c) / np.maximum(n, 1))
    co, no = C - c, N - n
    outside = xlogy(co, co / np.maximum(no, 1)) + xlogy(no - co, (no - co) / np.maximum(no, 1))
    null = xlogy(C, C / N) + xlogy(N - C, (N - C) / N)
    llr = inside + outside - null
    low = c * N < C * n
    return np.where(low, llr, 0.0)


def scan_cold_clusters(
    xy: np.ndarray,
    is_post: np.ndarray,
    radii_ft=(1320.0, 2640.0, 5280.0, 7920.0),
    n_sims: int = 999,
    seed: int = 4821,
    min_points: int = 20,
    grid_ft: float = 500.0,
    focus_xy: np.ndarray | None = None,
    focus_dist_ft: float = 1500.0,
    batch: int = 100,
) -> dict:
    """Circular scan for areas where construction-period severe crashes are
    *less* frequent than expected given the citywide pre/construction split.

    Null: the construction/pre label is exchangeable across crash locations
    (any citywide reduction is spatially uniform). Monte Carlo p-values come
    from permuting the labels. If ``focus_xy`` is supplied (points along the
    Colfax centerline), a second, focused p-value is computed for the
    maximum LLR over circles centered within ``focus_dist_ft`` of it.
    """
    rng = np.random.default_rng(seed)
    N = len(xy)
    C = int(is_post.sum())
    tree = cKDTree(xy)
    centers = np.unique(np.round(xy / grid_ft) * grid_ft, axis=0)
    rows, cols, meta = [], [], []
    for r in radii_ft:
        nbrs = tree.query_ball_point(centers, r)
        for ci, idx in enumerate(nbrs):
            if len(idx) < min_points:
                continue
            rows.extend([len(meta)] * len(idx))
            cols.extend(idx)
            meta.append((centers[ci, 0], centers[ci, 1], r))
    M = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(meta), N)
    )
    meta = np.array(meta)
    n_in = np.asarray(M.sum(axis=1)).ravel()
    c_in = M @ is_post.astype(np.float32)
    obs = bernoulli_llr(c_in, n_in, C, N)

    if focus_xy is not None:
        ftree = cKDTree(focus_xy)
        focus = ftree.query(meta[:, :2])[0] <= focus_dist_ft
    else:
        focus = np.zeros(len(meta), dtype=bool)

    max_all = np.empty(n_sims)
    max_focus = np.full(n_sims, np.nan)
    done = 0
    labels = is_post.astype(np.float32)
    while done < n_sims:
        b = min(batch, n_sims - done)
        perm = np.stack([rng.permutation(labels) for _ in range(b)], axis=1)
        c_sim = M @ perm
        llr = bernoulli_llr(c_sim, n_in[:, None], C, N)
        max_all[done : done + b] = llr.max(axis=0)
        if focus.any():
            max_focus[done : done + b] = llr[focus].max(axis=0)
        done += b

    best = int(obs.argmax())
    res = {
        "n_circles": len(meta),
        "n_points": N,
        "n_post": C,
        "best_llr": float(obs[best]),
        "best_center": (float(meta[best, 0]), float(meta[best, 1])),
        "best_radius_ft": float(meta[best, 2]),
        "best_c": float(c_in[best]),
        "best_n": float(n_in[best]),
        "best_expected": float(n_in[best] * C / N),
        "p_global": float((1 + (max_all >= obs.max()).sum()) / (n_sims + 1)),
        "circles": meta,
        "llr": obs,
        "c_in": np.asarray(c_in).ravel(),
        "n_in": n_in,
    }
    if focus.any():
        fbest = int(np.where(focus, obs, -np.inf).argmax())
        res.update(
            {
                "focus_best_llr": float(obs[fbest]),
                "focus_best_center": (float(meta[fbest, 0]), float(meta[fbest, 1])),
                "focus_best_radius_ft": float(meta[fbest, 2]),
                "focus_best_c": float(c_in[fbest]),
                "focus_best_n": float(n_in[fbest]),
                "focus_best_expected": float(n_in[fbest] * C / N),
                "p_focus": float((1 + (max_focus >= obs[focus].max()).sum()) / (n_sims + 1)),
            }
        )
    return res


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def prepare(buffer_ft: float = 250.0) -> tuple[pl.DataFrame, dict]:
    df = add_period(load_crashes())
    geom = corridor_geometry(df)
    df = zone_group(assign_zones(df, geom, buffer_ft=buffer_ft))
    return df, geom


def buffer_sensitivity(
    df: pl.DataFrame, geom: dict, buffers=(100.0, 150.0, 250.0, 300.0), outcome: str = "severe"
) -> pl.DataFrame:
    """Primary DiD re-estimated for several corridor buffer widths (feet).
    Colfax and the 14th Ave line are about 590 ft apart, so buffers above
    ~300 ft would start to overlap."""
    rows = []
    for b in buffers:
        d = zone_group(assign_zones(df.drop("zone", "group", strict=False), geom, buffer_ft=b))
        counts = yearly_counts(d, outcome)
        tab, _ = did_poisson(counts)
        for r in tab.iter_rows(named=True):
            rows.append(
                {"buffer_ft": b, "n_colfax": int(counts.filter(pl.col("group") == "colfax")["n"].sum())} | r
            )
    return pl.DataFrame(rows)


def run_primary(df: pl.DataFrame, geom: dict, n_sims: int = 999, outcome: str = "severe") -> dict:
    """Run every test for one outcome definition and return results in a dict."""
    out: dict = {"outcome": outcome}
    counts = yearly_counts(df, outcome)
    out["counts"] = counts
    out["counts_detail"] = yearly_counts(df, outcome, by="zone")
    out["did"], out["dispersion"] = did_poisson(counts)
    out["did_by_year"], _ = did_poisson(counts, by_year=True)
    out["did_no2020"], _ = did_poisson(counts, drop_years=(2020,))
    out["did_2021_on"], _ = did_poisson(counts, pre_years=range(2021, 2025))
    out["did_trend"] = did_poisson_trend(counts)
    try:
        out["did_nb"], out["nb_alpha"] = did_negbin(counts)
    except Exception as exc:  # NB fit can fail on sparse counts
        out["did_nb"], out["nb_alpha"] = pl.DataFrame(), float("nan")
        out["did_nb_error"] = str(exc)
    out["exact_conditional"] = pl.DataFrame(
        [
            exact_conditional_poisson(counts, g)
            | {"pre_period": "2013-2024"}
            for g in ("colfax", "alt")
        ]
        + [
            exact_conditional_poisson(counts, g, pre_years=range(2021, 2025))
            | {"pre_period": "2021-2024"}
            for g in ("colfax", "alt")
        ]
    )
    out["pretrend"] = pretrend_test(counts)
    out["fisher_colfax"] = exact_share_test(counts, "colfax")
    out["fisher_alt"] = exact_share_test(counts, "alt")

    c = counts.filter(pl.col("group") == "colfax")
    pre = int(c.filter(pl.col("year") < FIRST_POST_YEAR)["n"].sum())
    post = int(c.filter(pl.col("year") >= FIRST_POST_YEAR)["n"].sum())
    years_pre = int(c.filter(pl.col("year") < FIRST_POST_YEAR).height)
    years_post = int(c.filter(pl.col("year") >= FIRST_POST_YEAR).height)
    window_ft = geom["x_east"] - geom["x_west"]
    wins = placebo_windows(df, window_ft, outcome=outcome)
    stride = int(np.ceil(window_ft / 1320.0))
    out["placebo"] = placebo_test(wins, pre, post, years_pre, years_post, stride=stride)
    out["placebo_sens"] = pl.DataFrame(
        [
            {"baseline_ratio": r}
            | {k: v for k, v in placebo_test(wins, pre, post, years_pre, years_post, r, stride).items() if k != "placebo"}
            for r in (0.5, 0.25, 0.1)
        ]
    )

    sev = seasonal_window(df).filter(pl.col(outcome) & pl.col("x").is_not_null())
    xy = sev.select("x", "y").to_numpy()
    is_post = (sev.get_column("year").to_numpy() >= FIRST_POST_YEAR)
    xs = np.linspace(geom["x_west"], geom["x_east"], 200)
    s, b = geom["lines"]["colfax"]
    out["scan"] = scan_cold_clusters(
        xy, is_post, n_sims=n_sims, focus_xy=np.column_stack([xs, s * xs + b])
    )
    return out


# --------------------------------------------------------------------------
# Full-calendar version: seasonality handled in the model, not by a window
# --------------------------------------------------------------------------
def monthly_counts(df: pl.DataFrame, outcome: str = "severe", by: str = "group") -> pl.DataFrame:
    """Zone x calendar-month counts of outcome crashes for every month from the
    first observed month to Sep 2026 (complete grid, zeros filled).

    Nothing after Sep 27, 2026 is used (construction ended). ``days`` is the
    exposure of each cell, the number of days observed in that month (Sep 2026
    has 27), and is used as a Poisson offset by the monthly models.
    """
    d = df.filter(
        pl.col(by).is_not_null() & pl.col("dt").is_not_null() & (pl.col("dt") < FULLCAL_END_EXCL)
    ).with_columns(pl.col("dt").dt.truncate("1mo").dt.date().alias("ym"))
    first = d.get_column("ym").min()
    last = FULLCAL_END_EXCL.date().replace(day=1)
    months = pl.DataFrame({"ym": pl.date_range(first, last, "1mo", eager=True)})
    groups = pl.DataFrame({by: sorted(d.get_column(by).unique().to_list())})
    grid = groups.join(months, how="cross").with_columns(
        pl.col("ym").dt.year().alias("year"),
        pl.col("ym").dt.month().cast(pl.Int64).alias("month"),
        (
            pl.min_horizontal(pl.col("ym").dt.offset_by("1mo"), pl.lit(FULLCAL_END_EXCL.date()))
            - pl.col("ym")
        )
        .dt.total_days()
        .alias("days"),
    )
    n = d.filter(pl.col(outcome)).group_by(by, "ym").agg(pl.len().alias("n"))
    return (
        grid.join(n, on=[by, "ym"], how="left")
        .with_columns(pl.col("n").fill_null(0))
        .rename({by: "group"} if by != "group" else {})
        .sort("group", "ym")
    )


def _monthly_frame(
    counts: pl.DataFrame,
    drop_years=(),
    pre_start: date | None = None,
    post_start: date = CONSTRUCTION_START,
    max_month: int | None = None,
):
    """Model frame (pandas) for the monthly models.

    ``post`` is 1 from ``post_start``. If ``post_start`` is later than the real
    construction start, the months in between are dropped, not relabelled.
    ``max_month`` keeps only Jan..max_month (used to approximate the old window).
    """
    c = counts.filter(~pl.col("year").is_in(list(drop_years)))
    if pre_start is not None:
        c = c.filter(pl.col("ym") >= pre_start)
    if post_start != CONSTRUCTION_START:
        c = c.filter(~((pl.col("ym") >= CONSTRUCTION_START) & (pl.col("ym") < post_start)))
    if max_month is not None:
        c = c.filter(pl.col("month") <= max_month)
    ang = 2 * math.pi * (pl.col("month") - 0.5) / 12
    c = c.with_columns(
        (pl.col("ym") >= post_start).cast(pl.Int8).alias("post"),
        ang.sin().alias("sin1"),
        ang.cos().alias("cos1"),
        (pl.col("year") + (pl.col("month") - 0.5) / 12 - 2019.0).alias("t"),
        pl.col("ym").cast(pl.String).alias("ym_label"),
        pl.col("days").cast(pl.Float64).log().alias("log_days"),
    )
    for u in ("colfax", "alt", "corridor"):
        c = c.with_columns((pl.col("group") == u).cast(pl.Int8).alias(f"is_{u}")).with_columns(
            (pl.col(f"is_{u}") * pl.col("post")).alias(f"{u}_post"),
            (pl.col(f"is_{u}") * pl.col("sin1")).alias(f"{u}_sin"),
            (pl.col(f"is_{u}") * pl.col("cos1")).alias(f"{u}_cos"),
            (pl.col(f"is_{u}") * pl.col("t")).alias(f"{u}_t"),
        )
    return c.to_pandas()


SEASONALITY_FE = {
    "month": "C(year) + C(month)",  # shared month-of-year effects + year effects (primary)
    "time": "C(ym_label)",  # one effect per calendar month of the sample (most flexible)
    "zone_harmonic": "C(year) + C(month)",  # primary + zone-specific first-harmonic seasonality
    "none": "C(year)",  # no seasonal control (for contrast only)
}


def _did_monthly(
    counts: pl.DataFrame,
    units: list[str],
    seasonality: str = "month",
    trend: bool = False,
    **frame_kw,
) -> tuple[pl.DataFrame, float]:
    """Poisson difference-in-differences on zone x month counts with a
    log-exposure offset. ``seasonality`` picks how seasonal swings are removed
    (see ``SEASONALITY_FE``); the citywide (``rest``) trend and seasonality are
    absorbed by the fixed effects, so ``<unit>_post`` is the unit's rate ratio
    (construction / pre) relative to the rest of the city."""
    if seasonality not in SEASONALITY_FE:
        raise ValueError(f"seasonality must be one of {list(SEASONALITY_FE)}")
    df = _monthly_frame(counts, **frame_kw)
    rhs = [SEASONALITY_FE[seasonality]]
    rhs += [f"is_{u}" for u in units] + [f"{u}_post" for u in units]
    if seasonality == "zone_harmonic":
        rhs += [f"{u}_{h}" for u in units for h in ("sin", "cos")]
    if trend:
        rhs += [f"{u}_t" for u in units]
    fit, qfit, dispersion = _fit_glm("n ~ " + " + ".join(rhs), df, offset=df["log_days"])
    return _tidy(fit, qfit, [f"{u}_post" for u in units]), dispersion


def did_poisson_monthly(counts: pl.DataFrame, seasonality: str = "month", trend: bool = False, **kw):
    """Colfax and 13th/14th Ave vs the rest of the city (monthly counts from
    :func:`monthly_counts`). Returns (tidy table, Pearson dispersion)."""
    return _did_monthly(counts, ["colfax", "alt"], seasonality, trend, **kw)


def did_combined_monthly(counts: pl.DataFrame, seasonality: str = "month", trend: bool = False, **kw):
    """Combined corridor vs the rest of the city (``counts`` from
    :func:`combine_corridor` applied to monthly counts)."""
    return _did_monthly(counts, ["corridor"], seasonality, trend, **kw)


def seasonal_profile(df: pl.DataFrame, outcome: str = "severe") -> tuple[pl.DataFrame, dict]:
    """Citywide month-of-year profile of outcome crashes and a test that it is
    not flat. Poisson model ``n ~ C(year) + C(month)`` on citywide monthly
    counts with a log-days offset; ``relative_rate`` is exp(month effect)
    divided by the mean over months (1.0 = an average month). The likelihood-ratio
    test is scaled by the Pearson dispersion when that exceeds 1."""
    m = monthly_counts(df, outcome)
    c = m.group_by("ym", "year", "month", "days").agg(pl.col("n").sum()).sort("ym")
    d = c.with_columns(pl.col("days").cast(pl.Float64).log().alias("log_days")).to_pandas()
    kw = {"family": sm.families.Poisson(), "offset": d["log_days"].to_numpy()}
    full = smf.glm("n ~ C(year) + C(month)", d, **kw).fit()
    reduced = smf.glm("n ~ C(year)", d, **kw).fit()
    disp = max(float(full.pearson_chi2 / full.df_resid), 1.0)
    lr = float((reduced.deviance - full.deviance) / disp)
    df_lr = int(reduced.df_resid - full.df_resid)
    coefs = np.array([0.0] + [full.params[f"C(month)[T.{k}]"] for k in range(2, 13)])
    rel = np.exp(coefs) / np.exp(coefs).mean()
    profile = pl.DataFrame({"month": list(range(1, 13)), "relative_rate": rel})
    return profile, {"lr": lr, "df": df_lr, "p_value": float(stats.chi2.sf(lr, df_lr)), "dispersion": disp}


FULLCAL_SPECS = {
    "primary: year + month-of-year effects": dict(seasonality="month"),
    "month-of-sample effects": dict(seasonality="time"),
    "zone-specific seasonality (harmonics)": dict(seasonality="zone_harmonic"),
    "month-of-year + zone linear trends": dict(seasonality="month", trend=True),
    "excluding 2020": dict(seasonality="month", drop_years=(2020,)),
    "2021 to Sep 2024 baseline": dict(seasonality="month", pre_start=date(2021, 1, 1)),
    "Oct-Dec 2024 excluded (post from Jan 2025)": dict(seasonality="month", post_start=date(2025, 1, 1)),
    "Jan-Aug months only (approximates old window)": dict(seasonality="month", max_month=8),
    "no seasonal control (contrast)": dict(seasonality="none"),
}


def fullcal_specs(counts: pl.DataFrame, combined: bool = False) -> pl.DataFrame:
    """Rate ratios under every full-calendar specification. ``counts`` are
    monthly counts (Colfax / alt / rest), or, with ``combined=True``, monthly
    counts already merged by :func:`combine_corridor`."""
    fn = did_combined_monthly if combined else did_poisson_monthly
    rows = []
    for name, kw in FULLCAL_SPECS.items():
        tab, disp = fn(counts, **kw)
        for r in tab.iter_rows(named=True):
            rows.append({"spec": name, "dispersion": disp} | r)
    return pl.DataFrame(rows)


def fullcal_buffer_sensitivity(
    df: pl.DataFrame, geom: dict, buffers=(100.0, 150.0, 250.0, 300.0), outcome: str = "severe"
) -> pl.DataFrame:
    """Primary monthly DiD re-estimated for several corridor buffer widths (ft)."""
    rows = []
    for b in buffers:
        d = zone_group(assign_zones(df.drop("zone", "group", strict=False), geom, buffer_ft=b))
        counts = monthly_counts(d, outcome)
        tab, _ = did_poisson_monthly(counts)
        for r in tab.iter_rows(named=True):
            rows.append({"buffer_ft": b, "n_colfax": int(counts.filter(pl.col("group") == "colfax")["n"].sum())} | r)
    return pl.DataFrame(rows)


def run_fullcalendar(df: pl.DataFrame, outcome: str = "severe") -> dict:
    """Full-calendar analysis for one outcome. ``df`` has ``group`` assigned."""
    counts = monthly_counts(df, outcome)
    comb = combine_corridor(counts)
    profile, profile_test = seasonal_profile(df, outcome)
    return {
        "outcome": outcome,
        "counts": counts,
        "specs": fullcal_specs(counts),
        "combined_specs": fullcal_specs(comb, combined=True),
        "profile": profile,
        "profile_test": profile_test,
    }


def run_fullcalendar_wide(df_w: pl.DataFrame, outcome: str = "severe") -> dict:
    """Full-calendar analysis for the wide corridor (13th, 14th, Colfax, 16th,
    17th Ave merged). ``df_w`` comes from :func:`prepare_wide`."""
    counts = monthly_counts(df_w, outcome)
    comb = combine_corridor(counts)
    return {"outcome": outcome, "counts": comb, "combined_specs": fullcal_specs(comb, combined=True)}


def fullcal_summary(counts: pl.DataFrame) -> pl.DataFrame:
    """Observed crashes and days of exposure per group, pre vs construction."""
    return (
        counts.with_columns((pl.col("ym") >= CONSTRUCTION_START).alias("construction"))
        .group_by("group", "construction")
        .agg(pl.col("n").sum(), pl.col("days").sum())
        .with_columns((pl.col("n") / pl.col("days") * 365.25).round(2).alias("per_year"))
        .sort("group", "construction")
    )


def fig_seasonal_profile(profiles: dict[str, pl.DataFrame]):
    """Citywide month-of-year profile (1.0 = average month), one line per definition."""
    from plotnine import aes, geom_hline, geom_line, geom_point, ggplot, labs, scale_x_continuous, theme_minimal

    d = pl.concat([p.with_columns(pl.lit(k).alias("definition")) for k, p in profiles.items()])
    return (
        ggplot(d, aes("month", "relative_rate", color="definition"))
        + geom_hline(yintercept=1, linetype="dashed", color="grey")
        + geom_line(size=1)
        + geom_point(size=2)
        + scale_x_continuous(breaks=list(range(1, 13)))
        + labs(
            title="Citywide seasonal profile of severe crashes",
            subtitle="Relative rate per day after removing year effects (1.0 = average month)",
            x="Month",
            y="Relative rate",
            color=None,
        )
        + theme_minimal()
    )


def fig_fullcal_forest(specs: dict[str, pl.DataFrame]):
    """Rate ratios (95% CI) by specification for Colfax and 13th/14th Ave, one
    panel per outcome definition. ``specs`` maps a definition label to the
    ``specs`` table from :func:`fullcal_specs`."""
    from plotnine import (
        aes, element_text, facet_grid, geom_point, geom_segment, geom_vline, ggplot, labs, scale_x_log10, theme, theme_minimal,
    )

    d = pl.concat([t.with_columns(pl.lit(k).alias("definition")) for k, t in specs.items()]).with_columns(
        pl.col("term").replace({"colfax_post": "Colfax", "alt_post": "13th / 14th Ave"}).alias("unit")
    )
    order = list(FULLCAL_SPECS)[::-1]
    d = d.with_columns(pl.col("spec").cast(pl.Enum(order)))
    return (
        ggplot(d, aes(y="spec", x="rate_ratio"))
        + geom_vline(xintercept=1, linetype="dashed", color="grey")
        + geom_segment(aes(x="ci_low", xend="ci_high", yend="spec"))
        + geom_point(size=2)
        + facet_grid("definition ~ unit")
        + scale_x_log10()
        + labs(
            title="Full-calendar DiD rate ratios by specification",
            subtitle="Rate ratio vs the rest of Denver, construction vs pre",
            x="Rate ratio (log scale)",
            y="",
        )
        + theme_minimal()
        + theme(axis_text_y=element_text(size=7))
    )


def save_fullcal_outputs(fullcal: dict, out_dir: Path = OUT_DIR, fig_dir: Path = FIG_DIR) -> None:
    """Write the full-calendar tables and figures. ``fullcal`` maps an outcome
    name (``severe`` / ``severe_counts``) to :func:`run_fullcalendar` results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, r in fullcal.items():
        r["specs"].write_csv(out_dir / f"fullcal_did_{name}.csv")
        r["combined_specs"].write_csv(out_dir / f"fullcal_combined_{name}.csv")
        r["profile"].write_csv(out_dir / f"fullcal_profile_{name}.csv")
        if "wide" in r:
            r["wide"]["combined_specs"].write_csv(out_dir / f"fullcal_wide_{name}.csv")
    labels = {"severe": "A: offense code", "severe_counts": "B: count columns"}
    keep = [k for k in labels if k in fullcal]
    fig_seasonal_profile({labels[k]: fullcal[k]["profile"] for k in keep}).save(
        fig_dir / "seasonal_profile.png", width=8, height=4.5, dpi=150, verbose=False
    )
    fig_fullcal_forest({labels[k]: fullcal[k]["specs"] for k in keep}).save(
        fig_dir / "fullcal_forest.png", width=10, height=6, dpi=150, verbose=False
    )


# --------------------------------------------------------------------------
# Figures and tables
# --------------------------------------------------------------------------
GROUP_LABELS = {"colfax": "Colfax (Broadway-Colorado)", "alt": "13th / 14th Ave", "rest": "Rest of city"}


def fig_trend(counts: pl.DataFrame, label: str = ""):
    """Jan-Sep 27 severe crashes per zone, indexed to the 2013-2024 mean (=100)."""
    from plotnine import aes, annotate, geom_line, geom_point, geom_vline, ggplot, labs, scale_x_continuous, theme_minimal

    base = (
        counts.filter(pl.col("year") < FIRST_POST_YEAR)
        .group_by("group")
        .agg(pl.col("n").mean().alias("base"))
    )
    d = (
        counts.join(base, on="group")
        .with_columns(
            (100 * pl.col("n") / pl.col("base")).alias("index"),
            pl.col("group").replace(GROUP_LABELS).alias("zone"),
        )
    )
    return (
        ggplot(d, aes("year", "index", color="zone"))
        + geom_vline(xintercept=FIRST_POST_YEAR - 0.5, linetype="dashed", color="grey")
        + geom_line(size=1)
        + geom_point(size=2)
        + scale_x_continuous(breaks=list(range(2013, 2027, 1)))
        + labs(
            title=f"Severe crashes{label}, Jan 1 - Sep 27, indexed to each zone's 2013-2024 mean",
            subtitle="Dashed line: first construction-period year (2025)",
            x=None,
            y="Index (pre mean = 100)",
            color=None,
        )
        + theme_minimal()
    )


def fig_corridor_map(df: pl.DataFrame, geom: dict, outcome: str = "severe"):
    from plotnine import aes, coord_fixed, geom_point, geom_segment, ggplot, labs, scale_color_manual, theme_minimal

    pad = 2500
    d = (
        seasonal_window(df)
        .filter(pl.col(outcome) & pl.col("x").is_not_null())
        .filter(pl.col("x").is_between(geom["x_west"] - 800, geom["x_east"] + 800))
        .filter(pl.col("y").is_between(1690000 - pad, 1690000 + pad) | pl.col("group").is_in(["colfax", "alt"]))
        .with_columns(pl.col("period").replace({"pre": "2013-2024", "construction": "2025-2026"}))
    )
    xm = (geom["x_west"] + geom["x_east"]) / 2
    ymid = geom["lines"]["colfax"][0] * xm + geom["lines"]["colfax"][1]
    d = d.filter((pl.col("y") - ymid).abs() < 2500)
    segs = pl.DataFrame(
        {
            "street": list(geom["lines"]),
            "x": geom["x_west"],
            "xend": geom["x_east"],
            "y": [s * geom["x_west"] + b for s, b in geom["lines"].values()],
            "yend": [s * geom["x_east"] + b for s, b in geom["lines"].values()],
        }
    )
    return (
        ggplot()
        + geom_segment(segs, aes(x="x", xend="xend", y="y", yend="yend"), color="black", alpha=0.35)
        + geom_point(d, aes("x", "y", color="period"), size=1.6, alpha=0.7)
        + scale_color_manual(values={"2013-2024": "#9ecae1", "2025-2026": "#d94801"})
        + coord_fixed()
        + labs(
            title="Severe crashes near Colfax, Jan 1 - Sep 27",
            subtitle="Grey lines (top to bottom): Colfax, 14th Ave, 13th Ave centerlines, Broadway to Colorado Blvd",
            x="State Plane x (ft)",
            y="State Plane y (ft)",
            color="Period",
        )
        + theme_minimal()
    )


def fig_placebo(placebo: dict, label: str = ""):
    from plotnine import aes, geom_histogram, geom_vline, ggplot, labs, theme_minimal

    w = placebo["placebo"]
    return (
        ggplot(w, aes("lrr"))
        + geom_histogram(bins=30, fill="#9ecae1", color="white")
        + geom_vline(xintercept=placebo["observed_lrr"], color="#d94801", size=1.2)
        + labs(
            title=f"Colfax vs placebo corridors of the same length{label}",
            subtitle="Orange line: Colfax. x = log(construction rate / pre rate) of severe crashes",
            x="Log rate ratio (construction vs pre, per year)",
            y="Placebo windows",
        )
        + theme_minimal()
    )


def fig_scan(df: pl.DataFrame, geom: dict, scan: dict, outcome: str = "severe"):
    from plotnine import aes, coord_fixed, geom_path, geom_point, geom_segment, ggplot, labs, scale_color_manual, theme_minimal

    sev = seasonal_window(df).filter(pl.col(outcome) & pl.col("x").is_not_null())
    sev = sev.with_columns(pl.col("period").replace({"pre": "2013-2024", "construction": "2025-2026"}))
    th = np.linspace(0, 2 * np.pi, 100)

    def circ(cx, cy, r, name):
        return pl.DataFrame({"cx": cx + r * np.cos(th), "cy": cy + r * np.sin(th), "which": name})

    circles = pl.concat(
        [
            circ(*scan["best_center"], scan["best_radius_ft"], "Most extreme cold cluster (citywide)"),
            circ(*scan["focus_best_center"], scan["focus_best_radius_ft"], "Most extreme cold cluster near Colfax"),
        ]
    )
    s, b = geom["lines"]["colfax"]
    colfax = pl.DataFrame(
        {"x": [geom["x_west"]], "xend": [geom["x_east"]], "y": [s * geom["x_west"] + b], "yend": [s * geom["x_east"] + b]}
    )
    return (
        ggplot()
        + geom_point(sev, aes("x", "y", color="period"), size=0.5, alpha=0.5)
        + geom_segment(colfax, aes(x="x", xend="xend", y="y", yend="yend"), color="black", size=1.2)
        + geom_path(circles, aes("cx", "cy", linetype="which"), color="black")
        + scale_color_manual(values={"2013-2024": "#9ecae1", "2025-2026": "#d94801"})
        + coord_fixed()
        + labs(
            title="Spatial scan for areas with fewer construction-period severe crashes than expected",
            subtitle="Thick black line: Colfax study segment",
            x="State Plane x (ft)",
            y="State Plane y (ft)",
            color="Period",
            linetype=None,
        )
        + theme_minimal()
    )


def save_outputs(df: pl.DataFrame, geom: dict, results: dict, out_dir: Path = OUT_DIR, fig_dir: Path = FIG_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        r["counts"].pivot(on="group", index="year", values="n").write_csv(out_dir / f"counts_{name}.csv")
        r["did"].write_csv(out_dir / f"did_{name}.csv")
        r["did_trend"].write_csv(out_dir / f"did_trend_{name}.csv")
        r["placebo_sens"].write_csv(out_dir / f"placebo_{name}.csv")
    prim = results["severe"]
    for fname, fig, size in [
        ("trend.png", fig_trend(prim["counts"]), (9, 5)),
        ("corridor_map.png", fig_corridor_map(df, geom), (11, 5)),
        ("placebo.png", fig_placebo(prim["placebo"]), (8, 4.5)),
        ("scan.png", fig_scan(df, geom, prim["scan"]), (8, 7)),
    ]:
        fig.save(fig_dir / fname, width=size[0], height=size[1], dpi=150, verbose=False)


def main(n_sims: int = 999) -> dict:
    df, geom = prepare()
    results = {o: run_primary(df, geom, n_sims=n_sims, outcome=o) for o in ("severe", "severe_counts", "fatal_offense")}
    results["all_crashes"] = run_primary(
        df.with_columns(pl.lit(True).alias("all_crashes")), geom, n_sims=min(n_sims, 199), outcome="all_crashes"
    )
    save_outputs(df, geom, results)
    fullcal = {o: run_fullcalendar(df, o) for o in ("severe", "severe_counts")}
    df_w, _ = prepare_wide()
    for o, r in fullcal.items():
        r["wide"] = run_fullcalendar_wide(df_w, o)
    save_fullcal_outputs(fullcal)
    return {"df": df, "geom": geom, "results": results, "fullcal": fullcal}


if __name__ == "__main__":
    main()
