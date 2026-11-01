# colfax_construction_safety
Analysis to see if construction on Colfax improved safety


## Data

The raw data are not committed (`data/` is git-ignored, about 320 MB).

- **Source:** the "Traffic Accidents (Offenses)" dataset in the City and County of Denver Open Data Catalog: https://opendata-geospatialdenver.hub.arcgis.com/datasets/db00bd99ea534d8987e0913a191ebe19_325 (use the Download menu and choose CSV). The Vision Zero statistics page at https://denvergov.org/Government/Citywide-Programs-and-Initiatives/Vision-Zero/Statistics links to the same catalog.
- **Where to put it:** save the CSV in the `data/` folder at the project root (create the folder if it does not exist). The code reads the first `*.csv` it finds there, so keep a single CSV in that folder. The version used for the report was named `ODC_CRIME_TRAFFICACCIDENTS5YR_P_<id>.csv` and downloaded on 2026-10-29. The `.geodatabase` download is not needed.
- **Caution:** the dataset is updated Monday to Friday and records can be added, deleted, or changed, so a fresh download will not reproduce the report's numbers exactly. The catalog describes the dataset as covering the previous five calendar years plus the current year to date, but the file used here contains crashes from 2013 onward, and the analysis uses 2013-2024 as its pre-construction period. A newer download that starts later would change the pre period. Keep a copy of the downloaded file if you need exact reproducibility.

## Setup

```
uv sync                        # rebuild the pinned environment (.venv)
python -m pytest tests         # run the unit tests
cd report && quarto render colfax_report.qmd   # render the report (set QUARTO_PYTHON to .venv/bin/python if needed)
```

See the "Reproduce this analysis" section of the report for details.
