#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


DEFAULT_SOURCES = [
    ("non_function_regen_raw.jsonl", 42),
    ("function_regen_raw.jsonl", 42),
    ("non_function_retry_seed43.jsonl", 43),
    ("function_retry_seed43.jsonl", 43),
    ("non_function_retry_seed44.jsonl", 44),
    ("non_function_retry_seed45.jsonl", 45),
    ("non_function_retry_seed46.jsonl", 46),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Assemble final regenerated training data from raw and retry outputs."
        )
    )
    parser.add_argument(
        "--input-prompts",
        type=Path,
        default=Path("data_v2/prompts/train/all_train.jsonl"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("train_datasets/llama32_3b_instruct"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(
            "train_datasets/llama32_3b_instruct/all_train_regen.jsonl"
        ),
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path(
            "train_datasets/llama32_3b_instruct/assembly_report.json"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing final output and report.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Expected JSON object in {path}:{line_number}"
                )

            rows.append(row)

    return rows


def row_is_complete(row):
    generation = row.get("_generation")
    conversations = row.get("conversations")

    if not isinstance(generation, dict):
        return False

    turns = generation.get("turns")
    if not isinstance(turns, list) or not turns:
        return False

    if not all(turn.get("finish_reason") == "stop" for turn in turns):
        return False

    if not isinstance(conversations, list) or not conversations:
        return False

    final_message = conversations[-1]

    return (
        final_message.get("role") == "assistant"
        and isinstance(final_message.get("content"), str)
        and bool(final_message["content"].strip())
    )


def check_unique_ids(rows, path):
    ids = []

    for row in rows:
        sample_id = row.get("id")
        if not sample_id:
            raise RuntimeError(f"Missing sample ID in {path}")
        ids.append(sample_id)

    duplicates = [
        sample_id
        for sample_id, count in Counter(ids).items()
        if count > 1
    ]

    if duplicates:
        raise RuntimeError(
            f"Duplicate IDs in {path}: {duplicates[:10]}"
        )

    return ids


def write_jsonl_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    os.replace(temporary_path, path)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    args = parse_args()

    if not args.input_prompts.is_file():
        raise RuntimeError(
            f"Input prompts do not exist: {args.input_prompts}"
        )

    existing_outputs = [
        path
        for path in (args.output_file, args.report_file)
        if path.exists()
    ]

    if existing_outputs and not args.force:
        paths = "\n".join(f"  {path}" for path in existing_outputs)
        raise RuntimeError(
            "Refusing to overwrite existing final files.\n"
            "Review them, then rerun with --force:\n"
            f"{paths}"
        )

    original_rows = read_jsonl(args.input_prompts)
    original_ids = check_unique_ids(
        original_rows,
        args.input_prompts,
    )
    original_id_set = set(original_ids)

    selected = {}
    selected_source = {}
    source_report = []

    for filename, seed in DEFAULT_SOURCES:
        path = args.results_dir / filename

        if not path.exists():
            print(f"SKIP missing: {path}")
            source_report.append(
                {
                    "file": filename,
                    "seed": seed,
                    "status": "missing",
                    "rows": 0,
                    "accepted_stop_rows": 0,
                }
            )
            continue

        rows = read_jsonl(path)
        check_unique_ids(rows, path)

        accepted = 0
        ignored_incomplete = 0

        for row in rows:
            sample_id = row["id"]

            if sample_id not in original_id_set:
                raise RuntimeError(
                    f"Unexpected ID in {path}: {sample_id}"
                )

            if not row_is_complete(row):
                ignored_incomplete += 1
                continue

            generation = dict(row["_generation"])
            generation["selected_source"] = filename
            generation["seed"] = seed

            selected_row = dict(row)
            selected_row["_generation"] = generation

            # Later retry files overwrite earlier raw/retry results.
            selected[sample_id] = selected_row
            selected_source[sample_id] = filename
            accepted += 1

        print(
            f"{filename}: "
            f"rows={len(rows)}, "
            f"accepted_stop={accepted}, "
            f"ignored_incomplete={ignored_incomplete}"
        )

        source_report.append(
            {
                "file": filename,
                "seed": seed,
                "status": "loaded",
                "rows": len(rows),
                "accepted_stop_rows": accepted,
                "ignored_incomplete_rows": ignored_incomplete,
            }
        )

    missing_ids = [
        sample_id
        for sample_id in original_ids
        if sample_id not in selected
    ]

    if missing_ids:
        raise RuntimeError(
            f"Missing final results: {len(missing_ids)}; "
            f"first IDs: {missing_ids[:10]}"
        )

    unexpected_ids = sorted(set(selected) - original_id_set)

    if unexpected_ids:
        raise RuntimeError(
            f"Unexpected result IDs: {unexpected_ids[:10]}"
        )

    # Restore exactly the same order as the original all_train.jsonl.
    final_rows = [
        selected[sample_id]
        for sample_id in original_ids
    ]

    dataset_counts = Counter()
    source_counts = Counter()
    finish_reasons = Counter()

    for row in final_rows:
        if not row_is_complete(row):
            raise RuntimeError(
                f"Final validation failed for ID: {row.get('id')}"
            )

        dataset_counts[row["dataset"]] += 1
        source_counts[selected_source[row["id"]]] += 1

        for turn in row["_generation"]["turns"]:
            finish_reasons[turn["finish_reason"]] += 1

    if finish_reasons != Counter({"stop": len(final_rows)}):
        raise RuntimeError(
            f"Unexpected final finish reasons: {finish_reasons}"
        )

    write_jsonl_atomic(args.output_file, final_rows)

    output_sha256 = sha256_file(args.output_file)

    report = {
        "input_file": str(args.input_prompts),
        "output_file": str(args.output_file),
        "input_prompts": len(original_rows),
        "final_rows": len(final_rows),
        "missing_ids": 0,
        "unexpected_ids": 0,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "selected_source_counts": dict(sorted(source_counts.items())),
        "sources": source_report,
        "output_sha256": output_sha256,
    }

    args.report_file.parent.mkdir(parents=True, exist_ok=True)

    temporary_report = args.report_file.with_suffix(
        args.report_file.suffix + ".tmp"
    )
    temporary_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, args.report_file)

    print()
    print("Assembly completed.")
    print(f"Final rows: {len(final_rows)}")
    print(f"Finish reasons: {dict(finish_reasons)}")
    print(f"Output: {args.output_file}")
    print(f"Report: {args.report_file}")
    print(f"SHA-256: {output_sha256}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}")