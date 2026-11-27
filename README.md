# Bike sharing demand forecast

This project predicts total bike rentals (`cnt`) for the next five hours from the preceding 24 hours in the UCI Bike Sharing hourly dataset. It uses an LSTM encoder, Bahdanau attention, and an autoregressive LSTM decoder. The code also supports Luong attention and the daily CSV.

## Results

The hourly data has 17,379 records. We split it chronologically into 70% training, 15% validation, and 15% test data. Windows with a missing hour are excluded. This leaves 10,404 training, 2,603 validation, and 2,469 test forecast origins. The table reports the saved run on the held-out test period; MAE and RMSE are in rentals per hour.

| Forecast hour | R² / NSE | MAE | RMSE |
|---|---:|---:|---:|
| 1 | 0.9408 | 34.10 | 52.64 |
| 2 | 0.9152 | 40.77 | 62.90 |
| 3 | 0.9052 | 43.38 | 66.37 |
| 4 | 0.8947 | 45.97 | 69.96 |
| 5 | 0.8781 | 49.45 | 75.30 |
| Overall, pooled | **0.9069** | **42.73** | **65.87** |

Standard R² and NSE use the same formula here, so their values are identical. The full table, including squared Pearson correlation and macro averages, is in `outputs/metrics/test_metrics.csv`. These are results from one chronological split and one model seed.

## Data and model

The input contains historical `cnt`, calendar fields, observed weather fields, and sine/cosine encodings of hour, month, and weekday. The baseline drops historical `casual` and `registered` as redundant components of `cnt`; their past values would not inherently cause leakage. The date and row identifier are also excluded from model inputs. Scalers are fitted on training rows only. Validation and test windows may use preceding rows as historical context, but their target labels stay in their own splits. Windows crossing an unrecorded hour are excluded.

The encoder has two LSTM layers with 64 hidden units. Attention weights the 24 encoder states at each decoder step. The decoder begins with the last observed `cnt`; during validation and testing it feeds its own predictions into later steps. Training uses five-step MSE, AdamW, teacher forcing, gradient clipping, a learning-rate scheduler, and early stopping based on autoregressive validation loss.

The [EDA notebook](notebooks/EDA.ipynb) quantifies gaps, window loss, temporal dependence, feature associations, and split shift. The [compact EDA summary](report/EDA.md) can be regenerated with `python eda.py`. The [assignment report](report/Task%202%20Multi%20Step%20Bike%20Sharing%20Forecasting%20Report.md) gives more detail on the method and limitations.

## Project layout

| Path | Purpose |
|---|---|
| `data/` | Original UCI CSV files and dataset description. |
| `src/` | Data preparation, model, training, evaluation, and artifact code. |
| `main.py` | Command-line entry point for a training and evaluation run. |
| `notebooks/EDA.ipynb` | Interactive exploratory analysis. |
| `eda.py` | Rebuilds the short EDA summary and overview figure. |
| `report/` | Assignment report and scripts that generate report assets. |
| `tests/` | Pipeline regression checks. |
| `outputs/` | Saved checkpoints, metrics, and figures from example runs. |

## Environment and run

Use Python 3.13 and run commands from the repository root. On Windows, create the project environment and install its dependencies with `uv`:

```powershell
uv venv bike-forecast --python 3.13 --seed
uv pip install --python .\bike-forecast\Scripts\python.exe -r requirements-dev.txt
.\bike-forecast\Scripts\python.exe -m ipykernel install --user --name bike-forecast --display-name "Python (bike-forecast)"
```

If Python 3.13 and `pip` are already installed, `py -3.13 -m venv bike-forecast` and `.\bike-forecast\Scripts\python.exe -m pip install -r requirements-dev.txt` work too. The project's VS Code settings select this interpreter. In the EDA notebook, select **Python (bike-forecast)** as its kernel. `requirements.txt` contains runtime and notebook dependencies; `requirements-dev.txt` adds `nbformat` for the notebook generator.

Run the analysis, model, and checks with the environment's Python:

```powershell
.\bike-forecast\Scripts\python.exe eda.py
.\bike-forecast\Scripts\python.exe main.py
.\bike-forecast\Scripts\python.exe -m unittest discover -s tests -v
```

The source for `notebooks/EDA.ipynb` is `report/build_eda_notebook.py`. To rebuild it after editing that script, run `.\bike-forecast\Scripts\python.exe report/build_eda_notebook.py`; this replaces the notebook file, including saved cell outputs. Open the rebuilt notebook with the `Python (bike-forecast)` kernel before running it.

If VS Code remains stuck while connecting to the kernel, run `.\bike-forecast\Scripts\python.exe -m jupyterlab` from the repository root and open the local URL printed in the terminal. The server uses the same `bike-forecast` environment and can also be selected in VS Code through **Select Kernel → Existing Jupyter Server**.

For example, to try the daily file or a different attention mechanism:

```powershell
.\bike-forecast\Scripts\python.exe main.py --data-path data/day.csv --input-len 14 --output-dir outputs/day-new
.\bike-forecast\Scripts\python.exe main.py --attention luong --output-dir outputs/luong
```

`main.py --help` lists the other settings. A successful run saves model weights and accompanying JSON metadata under `outputs/checkpoints/`, metrics under `outputs/metrics/`, and diagnostic plots under `outputs/figures/`. The metadata records the feature order, scaler values, model settings, and hashes of the data and checkpoint. Use a separate output directory for each experiment to keep results paired with their checkpoint.

## Limits

The test period is later than training and has a different average rental count. The reported scores do not show how the model would perform on a new city or year. The simple baselines in the EDA provide context, but no ablation was run to measure the contribution of attention alone. Overlapping forecast windows also mean individual errors are not independent observations for significance tests. Rolling-origin validation across multiple cutoffs has not yet been run.

Data source: [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). The original data description is in `data/Readme.txt`.
