# Exploratory data analysis — hourly bike rentals

Source: `data/hour.csv`. The executable analysis is in [EDA.ipynb](../notebooks/EDA.ipynb). Run `python eda.py` to regenerate this compact summary and `outputs/figures/eda_overview.png`.

## Data quality and target

- 17,379 records from 2011-01-01 00:00 to 2012-12-31 23:00; 0 missing cells and 0 duplicate timestamps.
- 75 gaps between adjacent records, covering 165 absent hourly timestamps. The model excludes windows spanning a gap.
- `cnt = casual + registered` for all 17,379 rows. Counts range from 1 to 977; median 142, 95th percentile 563.1, mean 189.5.

## Patterns relevant to forecasting

- In training rows, mean demand peaks at 17:00 (390.9) and is lowest at 04:00 (5.2). This motivates hourly calendar features; it does not establish that 24 hours is the best history length.
- Training monthly means range from 94.4 to 218.5. Training temperature and humidity have Pearson correlations of 0.399 and -0.301 with `cnt`; these are associations, not causal effects.
- Chronological splits have different target levels:

| Split | Rows | Mean `cnt` | Median `cnt` |
|---|---:|---:|---:|
| Train | 12,165 | 159.9 | 119.0 |
| Validation | 2,607 | 281.0 | 251.0 |
| Test | 2,607 | 235.7 | 186.0 |

The shift means test accuracy should be interpreted for this later period, not assumed to transfer unchanged to another year.

## Preprocessing hypotheses and decisions

| Observation | Current choice | Evidence still needed |
|---|---|---|
| Absent hourly timestamps | Discard windows crossing a gap; never impute target labels. | Notebook measures gap lengths and window loss. Compare causal input imputation or masking on validation if needed. |
| Strong hourly pattern | 24-hour input and cyclic hour features are a baseline. | Compare 12, 24, 48, 72 and 168 hours on common validation target origins. |
| Calendar and weather associations | Retain historical calendar and observed weather. | Feature-group ablations; correlation alone does not establish predictive value. |
| Different target means by period | Split chronologically and fit scalers on training rows only. | Rolling-origin validation can test stability across several cutoffs. |
| `cnt = casual + registered` | Exclude historical components for simplicity. | Historical components do not inherently leak; compare with and without them on validation. Future components must never be inputs. |

The saved metrics use one chronological holdout. It protects the final test period but assesses only one cutoff. Rolling-origin evaluation has not been run; the notebook proposes expanding folds without presenting them as results.

## Simple test baselines

Both baselines use the same 2,469 complete test forecast origins and five forecast hours as the model. Persistence repeats the last observed count; seasonal persistence uses the count 24 hours before each forecast hour.

| Method | MAE | RMSE | Pooled R² |
|---|---:|---:|---:|
| Last observation | 166.89 | 234.95 | -0.1850 |
| Same hour previous day | 79.99 | 132.60 | 0.6226 |

The saved model results in `outputs/metrics/test_metrics.json` report MAE 42.73, RMSE 65.87, and pooled R² 0.9069. These baselines provide context; they do not isolate the effect of attention.

![Distribution, hourly and monthly patterns, and split means](../outputs/figures/eda_overview.png)
