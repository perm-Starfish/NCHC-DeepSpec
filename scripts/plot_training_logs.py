#!/usr/bin/env python3

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# epoch=10 step=388/420 loss=3.2996 | elapsed=309.6min | remaining=25.5min
KEY_VALUE_PATTERN = re.compile(
    r"epoch\s*=\s*(?P<epoch>\d+(?:\.\d+)?)"
    r".*?step\s*=\s*(?P<step>\d+)"
    r"(?:\s*/\s*(?P<total_steps>\d+))?"
    r".*?loss\s*=\s*(?P<loss>[-+0-9.eE]+)",
    re.IGNORECASE,
)

ELAPSED_PATTERN = re.compile(
    r"elapsed\s*=\s*(?P<elapsed>[-+0-9.eE]+)\s*min",
    re.IGNORECASE,
)

REMAINING_PATTERN = re.compile(
    r"remaining\s*=\s*(?P<remaining>[-+0-9.eE]+)\s*min",
    re.IGNORECASE,
)

LR_PATTERN = re.compile(
    r"(?:lr|learning_rate)\s*=\s*(?P<lr>[-+0-9.eE]+)",
    re.IGNORECASE,
)

GRAD_PATTERN = re.compile(
    r"(?:grad_norm|gradient_norm)\s*=\s*(?P<grad_norm>[-+0-9.eE]+)",
    re.IGNORECASE,
)


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_dictionary_line(line):
    """
    支援 Hugging Face Trainer 類型：
    {'loss': 3.2, 'grad_norm': 1.1, 'learning_rate': 1e-4, 'epoch': 2.0}
    """
    if "loss" not in line or "{" not in line or "}" not in line:
        return None

    start = line.find("{")
    end = line.rfind("}") + 1

    try:
        data = ast.literal_eval(line[start:end])
    except (ValueError, SyntaxError):
        return None

    if not isinstance(data, dict) or "loss" not in data:
        return None

    return {
        "epoch": safe_float(data.get("epoch")),
        "step": safe_int(data.get("step")),
        "total_steps": safe_int(data.get("total_steps")),
        "loss": safe_float(data.get("loss")),
        "elapsed_min": safe_float(data.get("elapsed_min")),
        "remaining_min": safe_float(data.get("remaining_min")),
        "learning_rate": safe_float(
            data.get("learning_rate", data.get("lr"))
        ),
        "grad_norm": safe_float(data.get("grad_norm")),
    }


def parse_log(log_path, model_name):
    rows = []
    fallback_step = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            match = KEY_VALUE_PATTERN.search(line)

            if match:
                elapsed_match = ELAPSED_PATTERN.search(line)
                remaining_match = REMAINING_PATTERN.search(line)
                lr_match = LR_PATTERN.search(line)
                grad_match = GRAD_PATTERN.search(line)

                row = {
                    "epoch": safe_float(match.group("epoch")),
                    "step": safe_int(match.group("step")),
                    "total_steps": safe_int(match.group("total_steps")),
                    "loss": safe_float(match.group("loss")),
                    "elapsed_min": safe_float(
                        elapsed_match.group("elapsed")
                        if elapsed_match else None
                    ),
                    "remaining_min": safe_float(
                        remaining_match.group("remaining")
                        if remaining_match else None
                    ),
                    "learning_rate": safe_float(
                        lr_match.group("lr") if lr_match else None
                    ),
                    "grad_norm": safe_float(
                        grad_match.group("grad_norm")
                        if grad_match else None
                    ),
                }
            else:
                row = parse_dictionary_line(line)

            if row is None or row["loss"] is None:
                continue

            fallback_step += 1

            if row["step"] is None:
                row["step"] = fallback_step

            row["model"] = model_name
            row["line_number"] = line_number
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No loss records were found in {log_path}. "
            "Please inspect the log format."
        )

    df = pd.DataFrame(rows)

    # resumed training 可能重複印出相同 step，保留最後一次紀錄
    df = (
        df.sort_values(["step", "line_number"])
        .drop_duplicates(subset=["step"], keep="last")
        .sort_values("step")
        .reset_index(drop=True)
    )

    maximum_step = df["step"].max()
    df["progress"] = df["step"] / maximum_step

    return df


def choose_smoothing_window(length, requested_window):
    if requested_window is not None:
        return max(1, min(requested_window, length))

    # 自動設定約為資料點數的 5%，範圍限制在 3～25
    return max(1, min(25, max(3, round(length * 0.05))))


def save_individual_loss_plot(df, output_dir, window):
    model = df["model"].iloc[0]
    smooth_column = f"loss_ma{window}"

    df[smooth_column] = (
        df["loss"]
        .rolling(window=window, min_periods=1, center=True)
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(
        df["step"],
        df["loss"],
        color="tab:blue",
        linewidth=0.8,
        alpha=0.28,
        label="Raw loss",
    )
    ax.plot(
        df["step"],
        df[smooth_column],
        color="tab:blue",
        linewidth=2.2,
        label=f"Moving average (window={window})",
    )

    ax.set_title(f"{model}: Training Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    filename = output_dir / f"{model.lower()}_training_loss.png"
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(all_df, output_dir):
    colors = {
        "DSpark": "tab:blue",
        "DFlash": "tab:orange",
        "EAGLE-3": "tab:green",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for model, df in all_df.groupby("model", sort=False):
        color = colors.get(model)
        window = choose_smoothing_window(len(df), None)
        smooth = (
            df["loss"]
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )

        axes[0].plot(
            df["step"],
            smooth,
            linewidth=2,
            color=color,
            label=f"{model} (MA={window})",
        )

        axes[1].plot(
            df["progress"] * 100,
            smooth,
            linewidth=2,
            color=color,
            label=model,
        )

    axes[0].set_title("Loss by Training Step")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Loss by Normalized Training Progress")
    axes[1].set_xlabel("Training progress (%)")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("DSpark vs. DFlash vs. EAGLE-3", fontsize=14)
    fig.tight_layout()

    fig.savefig(
        output_dir / "all_models_loss_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_elapsed_plot(all_df, output_dir):
    available = all_df.dropna(subset=["elapsed_min"])

    if available.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for model, df in available.groupby("model", sort=False):
        ax.plot(
            df["elapsed_min"],
            df["step"],
            linewidth=2,
            label=model,
        )

    ax.set_title("Training Progress over Time")
    ax.set_xlabel("Elapsed time (minutes)")
    ax.set_ylabel("Training step")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "all_models_training_speed.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_optional_metric_plot(all_df, metric, ylabel, filename, output_dir):
    available = all_df.dropna(subset=[metric])

    if available.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for model, df in available.groupby("model", sort=False):
        ax.plot(df["step"], df[metric], linewidth=1.7, label=model)

    ax.set_title(f"{ylabel} during Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(all_df):
    summaries = []

    for model, df in all_df.groupby("model", sort=False):
        first_n = min(10, len(df))
        last_n = min(10, len(df))

        initial_loss = df["loss"].iloc[:first_n].mean()
        final_loss = df["loss"].iloc[-last_n:].mean()
        reduction = (
            (initial_loss - final_loss) / initial_loss * 100
            if initial_loss != 0
            else float("nan")
        )

        summaries.append({
            "model": model,
            "records": len(df),
            "first_step": int(df["step"].min()),
            "last_step": int(df["step"].max()),
            "initial_loss_mean": initial_loss,
            "final_loss_mean": final_loss,
            "loss_reduction_percent": reduction,
            "minimum_loss": df["loss"].min(),
        })

    summary = pd.DataFrame(summaries)
    print("\nTraining summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot DSpark, DFlash, and EAGLE-3 training logs."
    )
    parser.add_argument("--dspark", type=Path, required=True)
    parser.add_argument("--dflash", type=Path, required=True)
    parser.add_argument("--eagle3", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_plots"),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Moving-average window for individual plots.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        ("DSpark", args.dspark),
        ("DFlash", args.dflash),
        ("EAGLE-3", args.eagle3),
    ]

    parsed = []

    for model_name, log_path in inputs:
        if not log_path.is_file():
            raise FileNotFoundError(f"{model_name} log not found: {log_path}")

        df = parse_log(log_path, model_name)
        window = choose_smoothing_window(len(df), args.smooth_window)

        save_individual_loss_plot(
            df.copy(),
            args.output_dir,
            window,
        )

        csv_path = args.output_dir / f"{model_name.lower()}_training_data.csv"
        df.to_csv(csv_path, index=False)
        parsed.append(df)

        print(
            f"{model_name}: parsed {len(df)} records "
            f"from {log_path}"
        )

    all_df = pd.concat(parsed, ignore_index=True)

    save_comparison_plot(all_df, args.output_dir)
    save_elapsed_plot(all_df, args.output_dir)

    save_optional_metric_plot(
        all_df,
        metric="learning_rate",
        ylabel="Learning rate",
        filename="all_models_learning_rate.png",
        output_dir=args.output_dir,
    )

    save_optional_metric_plot(
        all_df,
        metric="grad_norm",
        ylabel="Gradient norm",
        filename="all_models_gradient_norm.png",
        output_dir=args.output_dir,
    )

    all_df.to_csv(
        args.output_dir / "all_models_training_data.csv",
        index=False,
    )

    summary = print_summary(all_df)
    summary.to_csv(
        args.output_dir / "training_summary.csv",
        index=False,
    )

    print(f"\nPlots and CSV files saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
