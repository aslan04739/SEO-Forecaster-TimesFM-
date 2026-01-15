# SEO Forecast — Google Time Series

This workspace forecasts `Clicks` from the daily CSV using TimesFM when available, and falls back to a simple baseline if TimesFM weights are not accessible in your environment.

## Files
- predict_csv.py — reads `data.csv`, builds a forecast for the next 7 days, saves `resultat_prediction.csv`, and shows a plot.
- data.csv — source data. Expected columns include `Date` and `Clicks`.
- requirements.txt — pip dependencies for a plain virtualenv.

## Recommended Environment
TimesFM’s latest Python package requires JAX/PAXML for JAX backend or Torch weights for the Torch backend. The public Hugging Face repo `google/timesfm-1.0-200m` currently hosts JAX checkpoints. If JAX dependencies are unavailable, the script will automatically fall back to a simple baseline forecast.

For best results on macOS ARM:
- Use conda with Python 3.10 and install: `torch`, `jax`, `jaxlib`, `huggingface_hub`, `pandas`, `numpy`, `matplotlib`.

## Quick Start
Run inside the conda environment that has the needed packages:

```bash
/opt/anaconda3/bin/conda run -n timesfm_env python predict_csv.py
```

Or with the project virtualenv (if you prefer venv):

```bash
python -m venv .venv
"$(pwd)/.venv/bin/python" -m pip install -r requirements.txt
"$(pwd)/.venv/bin/python" predict_csv.py
```

## Output
- `resultat_prediction.csv` with two columns: `date`, `prediction` (next 7 days).
- A matplotlib chart showing the historical `Clicks` and the forecast.

## Notes
- Column detection is case-insensitive. The script looks for a `Date` column and the target column (default: `Clicks`).
- If TimesFM initialization or checkpoint loading fails, the script repeats the last 7 observed values as a simple baseline.
