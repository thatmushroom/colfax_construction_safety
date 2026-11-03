# colfax_construction_safety

Did construction on Colfax Avenue reduce serious injuries and deaths in Denver, or is the pattern just part of a citywide trend?

Construction ran from October 2024 to September 27, 2026 on Colfax between Broadway and Colorado Blvd. This project tests whether severe crashes (fatal or serious bodily injury, "SBI") on that stretch changed more than in the rest of the city, and whether crashes shifted onto parallel streets such as 13th and 14th Ave.

Full methods, tables and figures are in the report (`report/colfax_report.qmd`, rendered to HTML with Quarto) and the notebook (`colfax_analysis.ipynb`).

## The experiment

This is an observational before-and-after comparison (a difference-in-differences design), not a randomized experiment.

- **Outcome:** crashes with a fatality or serious bodily injury, counted per crash. Two definitions are reported side by side because they disagree:
  - **A, offense code:** the crash's top offense is FATAL or SBI. This reproduces the 318 SBI events quoted for 2025.
  - **B, count columns:** `SERIOUSLY_INJURED` or `FATALITIES` is greater than zero.
- **Seasonality control:** only January 1 to September 27 of each year is used, and no data after 2026-09-27.
- **Periods:** pre-construction is 2013-2024; construction is 2025-2026.
- **Treated unit:** Colfax between the Broadway and Colorado Blvd intersections (crashes within 250 ft of the centerline). Variants combine Colfax with 13th and 14th Ave, and with 13th, 14th, 16th and 17th Ave, as one corridor.
- **Comparison:** the rest of Denver, which absorbs any citywide change.
- **Tests:**
  - Poisson difference-in-differences on zone-by-year counts, which gives the Colfax rate ratio relative to the citywide change.
  - Negative-binomial and exact conditional Poisson checks, since there are only a handful of events.
  - Fisher exact test on Colfax's share of citywide severe crashes.
  - Placebo corridors of the same length on other streets, to guard against regression to the mean.
  - A spatial scan statistic for areas with fewer construction-period crashes than expected.
  - Sensitivity checks: buffer width, excluding 2020, a 2021-2024 baseline, zone-specific trends, leaving out each pre-period year, and all crashes as a comparison outcome.

## Results

A rate ratio below 1 means a unit fell more than the rest of the city.

| Comparison with the rest of Denver | A: offense code | B: count columns |
|---|---|---|
| Colfax alone | 0.38 (95% CI 0.19-0.79), p = 0.009 | 0.26 (0.09-0.70), p = 0.008 |
| 13th / 14th Ave alone | 1.5 (0.85-2.7), p = 0.16 | 2.0 (1.2-3.5), p = 0.013 |
| Colfax + 13th / 14th Ave combined | 0.75 (0.48-1.16), p = 0.19 | 0.85 (0.53-1.36), p = 0.51 |
| Wider corridor (13th, 14th, Colfax, 16th, 17th Ave) | 0.73 (0.48-1.10), p = 0.13 | 0.73 (0.47-1.13), p = 0.16 |

## Conclusions

- **Colfax alone:** severe crashes on the construction stretch fell roughly 60-75% relative to the rest of Denver, which is unlikely to be chance. The result holds under both definitions, across buffer widths, when the pre-period is restricted to 2021-2024, when any single pre-period year is dropped, and under exact small-count tests. Colfax also ranks at the low end of same-length placebo corridors, though that evidence weakens as the placebo set widens.
- **Not a citywide decline:** in the same months, severe crashes in the rest of the city were flat to slightly higher in 2025-2026 than in 2013-2024, and higher in 2026 than in 2025 (offense code: 276 to 321; count columns: 286 to 300). The premise that serious injuries and deaths are down in 2026 is not supported by this crash file.
- **Alternative routes:** crashes on 13th and 14th Ave fell less than the city, which is consistent with some crashes shifting there. It is statistically clear only under definition B and rests on 15-16 events, so it is suggestive, not established.
- **Net effect over the wider area:** when Colfax is combined with the parallel streets, the reduction is about 15-27% and is not statistically distinguishable from no change under either definition. The data fit a modest net benefit and fit no net benefit, and cannot separate real displacement from chance.
- **All crashes:** crashes of every severity fell on Colfax (about a one-third reduction) and across the wider corridor (about 17%), which would also be consistent with less traffic or closures during construction.

## Caveats

1. **The data do not agree on what counts as "severe."** The offense codes and the injury-count columns give different totals and trends (about 43% of SBI-coded crashes have a serious-injury count of zero), and neither reproduces the Denver Vision Zero dashboard's 94 deaths for 2025; the CSV shows 41 people killed. `FATALITIES` is blank or 0 for many FATAL-coded crashes. Counting each FATAL-coded crash as at least one death gives 87, 83, 70 and 81 for 2022-2025 against dashboard totals of 84, 83, 80 and 94: a close match for 2022-2023, but 10 and 13 short in 2024 and 2025 for reasons not yet known. Death and SBI totals should be confirmed with the city.
2. **There are very few events.** Colfax had 8 (A) or 4 (B) severe crashes in the construction period, so intervals are wide (the Colfax estimate is consistent with a drop of roughly 20% to 90%). Deaths alone are too few to test.
3. **This is not an experiment.** There is no traffic-volume data, so fewer cars or construction closures could explain a drop. Colfax counts peaked in 2021-2023 under definition A, so part of the drop may be a return to normal. The combined and wider corridors were chosen after seeing earlier results, so they are exploratory.

## Data

The raw data are not committed (`data/` is git-ignored, about 320 MB).

- **Source:** the "Traffic Accidents (Offenses)" dataset in the City and County of Denver Open Data Catalog: https://opendata-geospatialdenver.hub.arcgis.com/datasets/db00bd99ea534d8987e0913a191ebe19_325 (use the Download menu and choose CSV). The Vision Zero statistics page at https://denvergov.org/Government/Citywide-Programs-and-Initiatives/Vision-Zero/Statistics links to the same catalog.
- **Where to put it:** save the CSV in the `data/` folder at the project root (create the folder if it does not exist). The code reads the most recently modified `*.csv` in that folder (or the file named in the `COLFAX_CSV` environment variable). The version used for the current report is `ODC_CRIME_TRAFFICACCIDENTS5YR_P_-5915806814773269069(1).csv`, downloaded on 2026-11-02 (294,272 rows). It supersedes the 2026-10-29 download (294,140 rows), which lacked 132 records dated 2026-09-27 onward and had blank injury details on 48 recent records. Rerunning on it changed only minor figures (spatial-scan p-values and a few placebo-corridor counts); the main rate ratios are unchanged. The `.geodatabase` download is not needed.
- **Caution:** the dataset is updated Monday to Friday and records can be added, deleted, or changed, so a fresh download will not reproduce the report's numbers exactly. The catalog describes the dataset as covering the previous five calendar years plus the current year to date, but the file used here contains crashes from 2013 onward, and the analysis uses 2013-2024 as its pre-construction period. A newer download that starts later would change the pre period. Keep a copy of the downloaded file if you need exact reproducibility.

## Setup

```
uv sync                        # rebuild the pinned environment (.venv)
python -m pytest tests         # run the unit tests
cd report && quarto render colfax_report.qmd   # render the report (set QUARTO_PYTHON to .venv/bin/python if needed)
```

See the "Reproduce this analysis" section of the report for details.

## Project layout

| Path | Purpose |
|---|---|
| `analysis/colfax_analysis.py` | All data cleaning, zone assignment, tests and figure functions |
| `colfax_analysis.ipynb` | Notebook that runs the analysis and shows tables and figures |
| `report/colfax_report.qmd` | Quarto report with the full results |
| `tests/` | Unit tests for the helper functions |
| `pyproject.toml`, `uv.lock` | Pinned environment |
| `data/` | Raw crash data (not committed) |
