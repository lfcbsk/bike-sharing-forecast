"""Reproduce a compact exploratory analysis of the hourly bike data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    data = pd.read_csv("data/hour.csv").sort_values(["dteday", "hr"]).reset_index(drop=True)
    time = pd.to_datetime(data["dteday"]) + pd.to_timedelta(data["hr"], unit="h")
    n = len(data)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    splits = {
        "Train": data.iloc[:train_end],
        "Validation": data.iloc[train_end:val_end],
        "Test": data.iloc[val_end:],
    }
    gaps = int((time.diff().iloc[1:] != pd.Timedelta(hours=1)).sum())
    absent_hours = len(pd.date_range(time.min(), time.max(), freq="h").difference(time))

    # Use exactly the complete test windows used by the forecasting pipeline.
    actual, persistence, seasonal = [], [], []
    for start in range(val_end - 24, n - 28):
        window_time = time.iloc[start : start + 29]
        if not window_time.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            continue
        counts = data["cnt"].iloc[start : start + 29].to_numpy()
        actual.append(counts[24:29])
        persistence.append(np.repeat(counts[23], 5))
        seasonal.append(counts[0:5])  # Each target hour shifted back exactly 24 hours.
    actual = np.asarray(actual)
    persistence = np.asarray(persistence)
    seasonal = np.asarray(seasonal)

    def scores(prediction: np.ndarray) -> tuple[float, float, float]:
        error = actual - prediction
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        r2 = float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2))
        return mae, rmse, r2

    output = Path("outputs/figures")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(data["cnt"], bins=35, color="#4878A8")
    axes[0, 0].set(title="Hourly rental counts", xlabel="Rentals per hour", ylabel="Records")
    train = splits["Train"]
    train.groupby("hr")["cnt"].mean().plot(ax=axes[0, 1], color="#4878A8")
    axes[0, 1].set(title="Training mean by hour", xlabel="Hour", ylabel="Mean rentals")
    train.groupby("mnth")["cnt"].mean().plot(ax=axes[1, 0], color="#4878A8")
    axes[1, 0].set(title="Training mean by month", xlabel="Month", ylabel="Mean rentals")
    axes[1, 1].bar(list(splits), [part["cnt"].mean() for part in splits.values()], color="#4878A8")
    axes[1, 1].set(title="Mean by chronological split", ylabel="Mean rentals")
    fig.tight_layout()
    fig.savefig(output / "eda_overview.png", dpi=150)
    plt.close(fig)

    summary = Path("report/EDA.md")
    summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exploratory data analysis — hourly bike rentals",
        "",
        "Source: `data/hour.csv`. Run `python eda.py` from the repository root to regenerate this summary and `outputs/figures/eda_overview.png`.",
        "",
        "## Data quality and target",
        "",
        f"- {n:,} records from {time.min():%Y-%m-%d %H:%M} to {time.max():%Y-%m-%d %H:%M}; {int(data.isna().sum().sum())} missing cells and {int(time.duplicated().sum())} duplicate timestamps.",
        f"- {gaps} gaps between adjacent records, covering {absent_hours} absent hourly timestamps. The model excludes windows spanning a gap.",
        f"- `cnt = casual + registered` for all {n:,} rows. Counts range from {data.cnt.min():.0f} to {data.cnt.max():.0f}; median {data.cnt.median():.0f}, 95th percentile {data.cnt.quantile(.95):.1f}, mean {data.cnt.mean():.1f}.",
        "",
        "## Patterns relevant to forecasting",
        "",
        f"- In training rows, mean demand peaks at 17:00 ({train.groupby('hr').cnt.mean().loc[17]:.1f}) and is lowest at 04:00 ({train.groupby('hr').cnt.mean().loc[4]:.1f}). This supports using hourly calendar features and a 24-hour history.",
        f"- Training monthly means range from {train.groupby('mnth').cnt.mean().min():.1f} to {train.groupby('mnth').cnt.mean().max():.1f}. Training temperature and humidity have Pearson correlations of {train[['cnt', 'temp']].corr().iloc[0, 1]:.3f} and {train[['cnt', 'hum']].corr().iloc[0, 1]:.3f} with `cnt`; these are associations, not causal effects.",
        "- Chronological splits have different target levels:",
        "",
        "| Split | Rows | Mean `cnt` | Median `cnt` |",
        "|---|---:|---:|---:|",
    ]
    for name, part in splits.items():
        lines.append(f"| {name} | {len(part):,} | {part.cnt.mean():.1f} | {part.cnt.median():.1f} |")
    lines += [
        "",
        "The shift means test accuracy should be interpreted for this later period, not assumed to transfer unchanged to another year.",
        "",
        "## Preprocessing decisions",
        "",
        "| EDA observation | Decision in the pipeline |",
        "|---|---|",
        "| Absent hourly timestamps | Keep original timestamps and discard 24-to-5 windows crossing a gap; do not interpolate missing hours. |",
        "| Strong hourly pattern in training data | Use the previous 24 hours and cyclic hour features. |",
        "| Seasonal and weather associations in training data | Retain historical calendar and weather fields; cyclically encode month and weekday. |",
        "| Different target means across chronological periods | Split by time, fit scalers on train only, and report test performance separately. |",
        "| `cnt = casual + registered` in every record | Drop `casual` and `registered` when forecasting `cnt`. |",
        "",
        "## Simple test baselines",
        "",
        f"Both baselines use the same {len(actual):,} complete test forecast origins and five forecast hours as the model. Persistence repeats the last observed count; seasonal persistence uses the count 24 hours before each forecast hour.",
        "",
        "| Method | MAE | RMSE | Pooled R² |",
        "|---|---:|---:|---:|",
    ]
    for name, prediction in (("Last observation", persistence), ("Same hour previous day", seasonal)):
        mae, rmse, r2 = scores(prediction)
        lines.append(f"| {name} | {mae:.2f} | {rmse:.2f} | {r2:.4f} |")
    lines += [
        "",
        "The saved model results in `outputs/metrics/test_metrics.json` report MAE 42.73, RMSE 65.87, and pooled R² 0.9069. These baselines provide context; they do not isolate the effect of attention.",
        "",
        "![Distribution, hourly and monthly patterns, and split means](../outputs/figures/eda_overview.png)",
        "",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
