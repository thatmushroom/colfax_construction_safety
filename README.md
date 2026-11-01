# colfax_construction_safety
Analysis to see if construction on Colfax improved safety

## Setup

```
uv sync                        # rebuild the pinned environment (.venv)
python -m pytest tests         # run the unit tests
cd report && quarto render colfax_report.qmd   # render the report (set QUARTO_PYTHON to .venv/bin/python if needed)
```

See the "Reproduce this analysis" section of the report for details.
