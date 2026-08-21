#!/usr/bin/env python3
"""Plot training loss and checkpoint evaluation metrics in one figure.

Example:
    python scripts/plot_training_eval.py \
      --model-name DSpark \
      --training-log /data/logs/dspark_train.log \
      --eval 50=/data/logs/dspark_step50_eval.log \
      --eval 100=/data/logs/dspark_step100_eval.log \
      --eval 200=/data/logs/dspark_step200_eval.log \
      --eval 420=/data/logs/dspark_step420_eval.log \
      --dataset gsm8k \
      --output-dir training_plots/dspark_training_eval \
      --smooth-window 21
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRAIN_PATTERN = re.compile(
    r"epoch\s*=\s*(?P<epoch>\d+(?:\.\d+)?)"
    r".*?step\s*=\s*(?P<step>\d+)"
    r"(?:\s*/\s*(?P<total_steps>\d+))?"
    r".*?loss\s*=\s*(?P<loss>[-+0-9.eE]+)",
    re.IGNORECASE,
)

EVAL_METRICS = [
    "accept_len",
    "verify_rate",
    "accept_rate@0",
    "accept_rate@1",
    "accept_rate@2",
    "accept_rate@3",
    "accept_rate@4",
    "accept_rate@5",
    "accept_rate@6",
]


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "n/a", "-"}:
        return None
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if is_percent else number


def normalize_header(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("#", "num_")
    value = re.sub(r"[\s-]+", "_", value)
    aliases = {
        "accept_length": "accept_len",
        "accepted_length": "accept_len",
        "verification_rate": "verify_rate",
        "dataset_name": "dataset",
    }
    return aliases.get(value, value)


def split_pipe_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    cells = [cell.strip() for cell in stripped[1:-1].split("|")]
    if not cells or all(not cell for cell in cells):
        return None
    return cells


def parse_training_log(log_path: Path, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = TRAIN_PATTERN.search(line)
            if not match:
                continue
            rows.append(
                {
                    "model": model_name,
                    "epoch": float(match.group("epoch")),
                    "step": int(match.group("step")),
                    "total_steps": (
                        int(match.group("total_steps"))
                        if match.group("total_steps")
                        else None
                    ),
                    "loss": float(match.group("loss")),
                    "line_number": line_number,
                }
            )

    if not rows:
        raise RuntimeError(f"No training-loss records found in: {log_path}")

    frame = pd.DataFrame(rows)
    # Resumed jobs can print the same step more than once; keep the last record.
    frame = (
        frame.sort_values(["step", "line_number"])
        .drop_duplicates(subset=["step"], keep="last")
        .sort_values("step")
        .reset_index(drop=True)
    )
    return frame


def extract_pipe_tables(text: str) -> list[dict[str, str]]:
    """Extract rows from PrettyTable/Markdown-style evaluation tables."""
    lines = text.splitlines()
    extracted: list[dict[str, str]] = []

    for index, line in enumerate(lines):
        header_cells = split_pipe_row(line)
        if not header_cells:
            continue

        headers = [normalize_header(cell) for cell in header_cells]
        if "dataset" not in headers:
            continue
        if not any(metric in headers for metric in EVAL_METRICS):
            continue

        for candidate in lines[index + 1 :]:
            candidate_cells = split_pipe_row(candidate)
            if candidate_cells is None:
                # Table borders are skipped; an ordinary line ends this table.
                compact = candidate.strip()
                if not compact or set(compact) <= {"+", "-", "=", ":", " "}:
                    continue
                break
            if len(candidate_cells) != len(headers):
                continue
            normalized_candidate = [normalize_header(cell) for cell in candidate_cells]
            if normalized_candidate == headers:
                continue
            extracted.append(dict(zip(headers, candidate_cells)))

    return extracted


def parse_evaluation_log(
    log_path: Path,
    checkpoint_step: int,
    dataset: str,
    model_name: str,
) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = extract_pipe_tables(text)
    dataset_lower = dataset.strip().lower()
    matching = [
        row
        for row in rows
        if row.get("dataset", "").strip().lower() == dataset_lower
    ]

    if not matching:
        available = sorted(
            {
                row.get("dataset", "").strip()
                for row in rows
                if row.get("dataset", "").strip()
            }
        )
        raise RuntimeError(
            f"No evaluation row for dataset '{dataset}' in {log_path}. "
            f"Detected datasets: {available or 'none'}"
        )

    # A single log may contain several checkpoint rows. When possible, select
    # the row whose draft/checkpoint label contains the requested step.
    step_matching = []
    for row in matching:
        labels = " ".join(
            str(row.get(key, ""))
            for key in ("draft_model", "checkpoint", "step")
        )
        label_numbers = {int(number) for number in re.findall(r"\d+", labels)}
        if checkpoint_step in label_numbers:
            step_matching.append(row)
    if step_matching:
        matching = step_matching

    # Prefer the row containing the largest number of usable evaluation metrics.
    def row_score(row: dict[str, str]) -> int:
        return sum(safe_float(row.get(metric)) is not None for metric in EVAL_METRICS)

    selected = max(matching, key=row_score)
    parsed: dict[str, object] = {
        "model": model_name,
        "dataset": dataset,
        "checkpoint_step": checkpoint_step,
        "eval_log": str(log_path),
    }
    for metric in EVAL_METRICS:
        parsed[metric] = safe_float(selected.get(metric))

    if all(parsed[metric] is None for metric in EVAL_METRICS):
        raise RuntimeError(f"No supported evaluation metrics found in: {log_path}")

    return parsed


def parse_eval_argument(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--eval must use STEP=PATH format, for example "
            "--eval 100=/data/logs/dspark_step100_eval.log"
        )
    step_text, path_text = value.split("=", 1)
    try:
        step = int(step_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid checkpoint step in --eval: {step_text!r}"
        ) from error
    if step < 0 or not path_text.strip():
        raise argparse.ArgumentTypeError(f"Invalid --eval value: {value!r}")
    return step, Path(path_text)


def choose_window(length: int, requested: int | None) -> int:
    if requested is not None:
        return max(1, min(requested, length))
    return max(1, min(25, max(3, round(length * 0.05))))


def finite_values(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame.columns:
        return frame.iloc[0:0]
    return frame[pd.to_numeric(frame[column], errors="coerce").notna()]


def add_eval_step_guides(axes: list[plt.Axes], steps: list[int]) -> None:
    for axis in axes:
        for step in steps:
            axis.axvline(step, color="0.75", linewidth=0.8, alpha=0.35, zorder=0)


def plot_combined(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    output_path: Path,
    model_name: str,
    dataset: str,
    smooth_window: int,
    title: str | None,
) -> None:
    training = training.copy()
    evaluation = evaluation.sort_values("checkpoint_step").copy()
    training["smoothed_loss"] = (
        training["loss"]
        .rolling(window=smooth_window, min_periods=1, center=True)
        .mean()
    )

    fig, axes_array = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.15, 1.35]},
    )
    loss_ax, length_ax, rate_ax = axes_array
    axes = list(axes_array)
    eval_steps = evaluation["checkpoint_step"].astype(int).tolist()
    add_eval_step_guides(axes, eval_steps)

    loss_ax.plot(
        training["step"],
        training["loss"],
        color="tab:blue",
        linewidth=0.8,
        alpha=0.25,
        label="Raw training loss",
    )
    loss_ax.plot(
        training["step"],
        training["smoothed_loss"],
        color="tab:blue",
        linewidth=2.4,
        label=f"Training loss MA ({smooth_window})",
    )

    # Mark the smoothed training loss at each evaluated checkpoint.
    available_steps = training["step"].to_numpy()
    available_smooth = training["smoothed_loss"].to_numpy()
    marked_steps: list[int] = []
    marked_losses: list[float] = []
    for step in eval_steps:
        nearest_index = int(abs(available_steps - step).argmin())
        marked_steps.append(int(available_steps[nearest_index]))
        marked_losses.append(float(available_smooth[nearest_index]))
    loss_ax.scatter(
        marked_steps,
        marked_losses,
        color="black",
        s=28,
        zorder=5,
        label="Evaluated checkpoints",
    )
    loss_ax.set_ylabel("Training loss")
    loss_ax.set_title("Training objective")
    loss_ax.legend(loc="best")

    accept_len = finite_values(evaluation, "accept_len")
    if not accept_len.empty:
        length_ax.plot(
            accept_len["checkpoint_step"],
            accept_len["accept_len"],
            color="tab:orange",
            marker="o",
            markersize=6,
            linewidth=2,
            linestyle="--",
            label="Accept length",
        )
        length_ax.axhline(
            1.0,
            color="tab:red",
            linewidth=1.2,
            linestyle=":",
            label="Bonus-token baseline (1.0)",
        )
    else:
        length_ax.text(
            0.5,
            0.5,
            "accept_len unavailable",
            transform=length_ax.transAxes,
            ha="center",
            va="center",
        )
    length_ax.set_ylabel("Accepted tokens")
    length_ax.set_title(f"Checkpoint evaluation: {dataset}")
    length_ax.legend(loc="best")

    rate_styles = {
        "verify_rate": ("Verify rate", "tab:green", "o"),
        "accept_rate@0": ("Accept rate @0", "tab:purple", "s"),
        "accept_rate@1": ("Accept rate @1", "tab:brown", "^"),
    }
    plotted_rate = False
    for column, (label, color, marker) in rate_styles.items():
        available = finite_values(evaluation, column)
        if available.empty:
            continue
        rate_ax.plot(
            available["checkpoint_step"],
            available[column],
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=1.8,
            linestyle="--",
            label=label,
        )
        plotted_rate = True

    if not plotted_rate:
        rate_ax.text(
            0.5,
            0.5,
            "Rate metrics unavailable",
            transform=rate_ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        rate_ax.legend(loc="best")
    rate_ax.set_ylabel("Rate")
    rate_ax.set_xlabel("Training / checkpoint step")
    rate_ax.set_title("Verification and positional acceptance")

    for axis in axes:
        axis.grid(alpha=0.22)

    maximum_step = max(
        int(training["step"].max()),
        int(evaluation["checkpoint_step"].max()),
    )
    rate_ax.set_xlim(left=0, right=maximum_step * 1.02)

    figure_title = title or f"{model_name}: Training Loss vs. Evaluation Acceptance"
    fig.suptitle(figure_title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_acceptance_by_position(
    evaluation: pd.DataFrame,
    output_path: Path,
    model_name: str,
    dataset: str,
) -> bool:
    columns = [f"accept_rate@{index}" for index in range(7)]
    available_columns = [
        column
        for column in columns
        if column in evaluation.columns and evaluation[column].notna().any()
    ]
    if not available_columns:
        return False

    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = [int(column.split("@", 1)[1]) for column in available_columns]

    for _, row in evaluation.sort_values("checkpoint_step").iterrows():
        values = [row.get(column) for column in available_columns]
        numeric = [float(value) if pd.notna(value) else math.nan for value in values]
        ax.plot(
            positions,
            numeric,
            marker="o",
            linewidth=1.8,
            label=f"step {int(row['checkpoint_step'])}",
        )

    ax.set_title(f"{model_name}: Positional Acceptance on {dataset}")
    ax.set_xlabel("Draft-token position")
    ax.set_ylabel("Acceptance rate")
    ax.set_xticks(positions)
    ax.grid(alpha=0.25)
    ax.legend(title="Checkpoint", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine a training-loss curve with checkpoint evaluation metrics "
            "for one speculative-decoding method."
        )
    )
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument(
        "--eval",
        dest="evaluations",
        action="append",
        type=parse_eval_argument,
        required=True,
        metavar="STEP=PATH",
        help="Repeat once per evaluated checkpoint.",
    )
    parser.add_argument("--model-name", default="DSpark")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_eval_plots"),
    )
    parser.add_argument("--smooth-window", type=int, default=21)
    parser.add_argument("--title", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.training_log.is_file():
        raise FileNotFoundError(f"Training log not found: {args.training_log}")
    for step, eval_path in args.evaluations:
        if not eval_path.is_file():
            raise FileNotFoundError(
                f"Evaluation log for checkpoint {step} not found: {eval_path}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = parse_training_log(args.training_log, args.model_name)
    window = choose_window(len(training), args.smooth_window)

    evaluation_rows = [
        parse_evaluation_log(
            log_path=eval_path,
            checkpoint_step=step,
            dataset=args.dataset,
            model_name=args.model_name,
        )
        for step, eval_path in args.evaluations
    ]
    evaluation = (
        pd.DataFrame(evaluation_rows)
        .sort_values("checkpoint_step")
        .drop_duplicates(subset=["checkpoint_step"], keep="last")
        .reset_index(drop=True)
    )

    training_csv = args.output_dir / "training_data.csv"
    evaluation_csv = args.output_dir / "evaluation_data.csv"
    combined_png = args.output_dir / "training_vs_evaluation.png"
    positional_png = args.output_dir / "acceptance_by_position.png"

    training.to_csv(training_csv, index=False)
    evaluation.to_csv(evaluation_csv, index=False)
    plot_combined(
        training=training,
        evaluation=evaluation,
        output_path=combined_png,
        model_name=args.model_name,
        dataset=args.dataset,
        smooth_window=window,
        title=args.title,
    )
    positional_created = plot_acceptance_by_position(
        evaluation=evaluation,
        output_path=positional_png,
        model_name=args.model_name,
        dataset=args.dataset,
    )

    print(f"Parsed {len(training)} training-loss records.")
    print(f"Parsed {len(evaluation)} evaluated checkpoints.")
    print("\nEvaluation data:")
    display_columns = [
        column
        for column in ["checkpoint_step", *EVAL_METRICS]
        if column in evaluation.columns
    ]
    print(evaluation[display_columns].to_string(index=False))
    print(f"\nCombined figure: {combined_png.resolve()}")
    if positional_created:
        print(f"Position figure: {positional_png.resolve()}")
    print(f"CSV files: {training_csv.resolve()}, {evaluation_csv.resolve()}")


if __name__ == "__main__":
    main()
