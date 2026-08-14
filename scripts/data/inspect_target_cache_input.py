#!/usr/bin/env python3

import argparse
import json

from transformers import AutoConfig, AutoTokenizer

from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.parser import preprocess_record
from deepspec.data.target_cache_dataset import expected_target_cache_tensor_nbytes
from deepspec.utils import load_config, parse_opts_to_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument("--train-data-path", action="append", required=True)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    return parser.parse_args()


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def main():
    args = parse_args()
    config = parse_opts_to_config(args.opts, load_config(args.config))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target_model_name_or_path,
        local_files_only=bool(config.model.get("local_files_only", False)),
    )
    dataset = JsonLineDataset(data_paths=args.train_data_path)
    model_config = AutoConfig.from_pretrained(
        config.model.target_model_name_or_path,
        local_files_only=bool(config.model.get("local_files_only", False)),
    )
    hidden_size = int(model_config.hidden_size)
    num_hidden_layers = int(model_config.num_hidden_layers)
    invalid_layer_ids = [
        int(layer_id)
        for layer_id in config.model.target_layer_ids
        if int(layer_id) < -1 or int(layer_id) >= num_hidden_layers
    ]
    if invalid_layer_ids:
        raise ValueError(
            f"Invalid target_layer_ids={invalid_layer_ids} for a model with "
            f"{num_hidden_layers} decoder layers."
        )

    lengths = []
    loss_lengths = []
    skipped = 0
    total_bytes = 0
    for index in range(len(dataset)):
        processed = preprocess_record(
            dataset[index],
            tokenizer=tokenizer,
            chat_template=config.data.chat_template,
            max_length=int(config.data.max_length),
        )
        seq_len = int(processed["attention_mask"].sum().item())
        loss_len = int(processed["loss_mask"].sum().item())
        if loss_len < int(args.min_loss_tokens):
            skipped += 1
            continue
        lengths.append(seq_len)
        loss_lengths.append(loss_len)
        tensor_bytes = expected_target_cache_tensor_nbytes(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_target_layers=len(config.model.target_layer_ids),
        )
        total_bytes += sum(tensor_bytes.values())
        if (index + 1) % 1000 == 0 or index + 1 == len(dataset):
            print(f"[inspect] {index + 1}/{len(dataset)}", flush=True)

    lengths_sorted = sorted(lengths)

    def percentile(percent: float):
        if not lengths_sorted:
            return None
        offset = round((len(lengths_sorted) - 1) * percent)
        return lengths_sorted[offset]

    report = {
        "input_samples": len(dataset),
        "valid_samples": len(lengths),
        "skipped_below_min_loss_tokens": skipped,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "target_layer_ids": list(config.model.target_layer_ids),
        "total_tokens": sum(lengths),
        "mean_tokens": (sum(lengths) / len(lengths)) if lengths else None,
        "p50_tokens": percentile(0.50),
        "p90_tokens": percentile(0.90),
        "p95_tokens": percentile(0.95),
        "p99_tokens": percentile(0.99),
        "max_tokens": max(lengths) if lengths else None,
        "total_loss_tokens": sum(loss_lengths),
        "estimated_cache_bytes": total_bytes,
        "estimated_cache_size": human_bytes(total_bytes),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    dataset.close()


if __name__ == "__main__":
    main()
