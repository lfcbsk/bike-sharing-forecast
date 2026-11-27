"""Run attention ablations with independent seeds and summarize saved test metrics."""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ATTENTION_TYPES = ("bahdanau", "luong", "none", "mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--data-path", type=Path, default=Path("data/hour.csv"))
    parser.add_argument("--input-len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--attention-types", nargs="+", choices=ATTENTION_TYPES, default=list(ATTENTION_TYPES))
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs with saved metrics")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training")
    return parser.parse_args()


def metric_values(report: dict) -> dict[str, float]:
    values = {name: float(report["overall"][name]) for name in ("MAE", "RMSE", "R2")}
    values.update({f"step_{step}_MAE": float(report["per_step"][f"step_{step}"]["MAE"])
                   for step in range(1, 6)})
    return values


def validation_selection(run_dir: Path) -> dict[str, float | int]:
    """Persist the selected epoch and rounded validation MSE for reproducible summaries."""
    selection_path = run_dir / "validation_selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    log_path = run_dir / "run.log"
    pattern = re.compile(r"^Epoch (\d+)/\d+ .*Val MSE: ([0-9.]+) \* \[Best saved\]$", re.MULTILINE)
    matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No selected validation checkpoint in {log_path}")
    epoch, mse = matches[-1]
    selection = {"best_epoch": int(epoch), "best_val_mse": float(mse)}
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection


def summarize(rows: list[dict], output_dir: Path) -> None:
    import statistics

    metric_names = ("best_val_mse", "best_epoch", "MAE", "RMSE", "R2",
                    *(f"step_{step}_MAE" for step in range(1, 6)))
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("attention", "seed", *metric_names))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ("attention", "n", *(f"{name}_{stat}" for name in metric_names for stat in ("mean", "std")))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for attention in ATTENTION_TYPES:
            group = [row for row in rows if row["attention"] == attention]
            if not group:
                continue
            summary = {"attention": attention, "n": len(group)}
            for name in metric_names:
                values = [row[name] for row in group]
                summary[f"{name}_mean"] = statistics.mean(values)
                summary[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            writer.writerow(summary)


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    rows = []
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    for attention in args.attention_types:
        for seed in args.seeds:
            run_dir = output_dir / f"{attention}_s{seed}"
            metrics_path = run_dir / "metrics" / "test_metrics.json"
            command = [sys.executable, str(ROOT / "main.py"),
                       "--attention", attention, "--seed", str(seed),
                       "--data-path", str(args.data_path),
                       "--input-len", str(args.input_len),
                       "--epochs", str(args.epochs),
                       "--batch-size", str(args.batch_size),
                       "--output-dir", str(run_dir)]
            print(f"[{attention}, seed={seed}] {' '.join(command)}", flush=True)
            if args.dry_run:
                continue
            if not (args.resume and metrics_path.exists()):
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / "run.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            metadata_path = run_dir / "checkpoints" / "best_seq2seq_attention.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (metadata["seed"] != seed or
                    metadata["model"]["attention_type"] != attention or
                    metadata["preprocessing"]["input_len"] != args.input_len):
                raise ValueError(f"Saved run has different settings: {run_dir}")
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({"attention": attention, "seed": seed,
                         **validation_selection(run_dir), **metric_values(report)})
            summarize(rows, output_dir)
    if not args.dry_run:
        print(f"Saved {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
