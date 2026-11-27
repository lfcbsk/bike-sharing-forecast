# Bike sharing demand forecast

This project predicts total bike rentals (`cnt`) for the next five hours from the preceding 24 hours in the UCI Bike Sharing hourly dataset. It uses an LSTM encoder, Bahdanau attention, and an autoregressive LSTM decoder. An optional general multiplicative score is available under `--attention luong`; its query uses the preceding decoder state, so this is not a canonical Luong decoder. The code can also read the daily CSV.

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

This table describes the original single-seed Bahdanau checkpoint, which remains the main submitted model. The four-variant, three-seed ablation is reported below.

Standard R² and NSE use the same formula here, so their values are identical. The full table, including squared Pearson correlation and macro averages, is in `outputs/metrics/test_metrics.csv`. These are results from one chronological split and one model seed.

An attention ablation retrained each variant with seeds 42, 43, and 44 on the same 24-hour input and test split. Values are mean ± sample standard deviation across seeds, computed from [saved runs](outputs/ablation/summary.csv):

| Decoder context | Pooled MAE | Pooled RMSE | Pooled R² | Best validation MSE |
|---|---:|---:|---:|---:|
| Zero context (`none`) | 45.52 ± 2.68 | 70.60 ± 2.91 | 0.8929 ± 0.0087 | 0.2165 ± 0.0217 |
| Fixed mean context (`mean`) | 43.48 ± 1.67 | 67.85 ± 2.53 | 0.9011 ± 0.0074 | 0.2100 ± 0.0275 |
| Bahdanau | 43.99 ± 1.53 | 67.56 ± 1.82 | 0.9020 ± 0.0053 | 0.1885 ± 0.0113 |
| General score (`luong`) | **41.74 ± 0.52** | **65.22 ± 0.31** | **0.9087 ± 0.0009** | **0.1841 ± 0.0090** |

The general-score variant has the lowest mean MAE and the lowest mean best-validation MSE (0.1841, versus 0.1885 for Bahdanau, 0.2100 for fixed mean, and 0.2165 for zero context). Its test advantage is a trend across only three seeds, not a significance claim. Fixed mean improves mean test MAE relative to zero context, while Bahdanau does not clearly improve on fixed mean; this suggests that having context matters more than learning nonuniform Bahdanau weights for this 24-hour input. The ablation is exploratory because all variants have already been evaluated on the held-out test period. A future final configuration must be selected on validation data before inspecting a fresh test set.

| Seed | Zero context MAE | Fixed mean MAE | Bahdanau MAE | General score MAE |
|---:|---:|---:|---:|---:|
| 42 | 42.73 | 42.27 | 42.73 | 41.98 |
| 43 | 48.06 | 45.38 | 43.54 | 42.09 |
| 44 | 45.77 | 42.78 | 45.69 | 41.13 |

| Decoder context | MAE at t+1 | MAE at t+5 |
|---|---:|---:|
| Zero context | 35.14 ± 3.63 | 52.70 ± 1.67 |
| Fixed mean context | 33.73 ± 0.77 | 50.38 ± 3.18 |
| Bahdanau | 34.38 ± 1.93 | 50.64 ± 1.08 |
| General score | 33.31 ± 1.32 | 47.33 ± 0.36 |

The mean MAE gap between general score and zero context grows from 1.82 at t+1 to 5.38 at t+5, consistent with a context path reducing autoregressive error accumulation. This pattern alone does not prove the mechanism. Validation and test rankings can disagree: at seed 42, zero context had worse best-validation MSE than Bahdanau seed 43 (0.2210 versus 0.1945), yet lower test MAE (42.73 versus 43.54). Across the 12 runs, best epochs were 4–10 while the 35-epoch teacher-forcing schedule was still at roughly 0.37–0.46.

Re-running Bahdanau with seed 42 gave MAE 42.7295, compared with 42.7329 in the previously saved checkpoint (difference 0.0034). The runs are close but not bitwise identical; the earlier environment and deterministic-algorithm settings were not recorded in enough detail to assign a single cause.

## Data and model

The input contains historical `cnt`, calendar fields, observed weather fields, and sine/cosine encodings of hour, month, and weekday. The baseline drops historical `casual` and `registered` as redundant components of `cnt`; their past values would not inherently cause leakage. The date and row identifier are also excluded from model inputs. Scalers are fitted on training rows only. Validation and test windows may use preceding rows as historical context, but their target labels stay in their own splits. Windows crossing an unrecorded hour are excluded.

The encoder has two LSTM layers with 64 hidden units. Attention weights the 24 encoder states at each decoder step. The decoder begins with the last observed `cnt`; during validation and testing it feeds its own predictions into later steps. Training uses five-step MSE, AdamW, teacher forcing, gradient clipping, a learning-rate scheduler, and early stopping based on autoregressive validation loss.

The [EDA notebook](notebooks/EDA.ipynb) quantifies gaps, window loss, temporal dependence, feature associations, and split shift. The [compact EDA summary](report/EDA.md) can be regenerated with `python eda.py`. The [assignment report](report/Task%202%20Multi%20Step%20Bike%20Sharing%20Forecasting%20Report.md) gives more detail on the method and limitations.

The same-hour previous-week baseline scores MAE 64.21 on 2,414 test windows with an observed hour exactly 168 hours earlier. The saved Bahdanau checkpoint scores MAE 43.17 on that same subset. The remaining 55 test windows lack at least one prior-week hour. Average attention weights from the original Bahdanau checkpoint are near uniform; this does not imply the context has no predictive value.

To compare context and attention mechanisms with the same pipeline, run `.\bike-forecast\Scripts\python.exe scripts/run_ablation.py`. It trains `bahdanau`, `luong`, `none`, and `mean` with seeds 42, 43, and 44 in separate `outputs/ablation/` directories, then writes `runs.csv` and `summary.csv`. Use `--resume` to continue after an interrupted run. The `none` variant supplies zero context; `mean` supplies the fixed average of encoder states. Both return uniform placeholder weights for the plotting interface.

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

The test period is later than training and has a different average rental count. The reported scores do not show how the model would perform on a new city or year. The ablation uses three seeds and one 24-hour input length; a larger seed set and 168-hour input comparison have not been run. The decoder also does not receive known calendar features for the forecast hours. Overlapping forecast windows mean individual errors are not independent observations for significance tests. Rolling-origin validation across multiple cutoffs has not yet been run. The older `outputs/day/` run has poor saved accuracy (R² 0.1668, MAE 1,497.48 rentals/day) and no checkpoint metadata, so it is not treated as a reproducible result of the current pipeline.

Data source: [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). The original data description is in `data/Readme.txt`.
