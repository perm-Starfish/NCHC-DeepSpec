#!/usr/bin/env python3
"""Build cleaned DeepSpec training/evaluation datasets from private JSONL files.

The script is deterministic: identical input bytes, seed, and arguments produce
identical splits and output ordering. It never modifies files under raw/.

Expected input filenames:
  cmath_zh_tw.jsonl
  gaokao-mathcloze_zh_tw.jsonl
  gaokao-mathqa_zh_tw.jsonl
  gsm8k_zh_tw.jsonl
  math_zh_tw.jsonl
  minif2f_zh_tw.jsonl
  mmlu_stem_zh_tw.jsonl
  ocw_zh_tw.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DATASET_FILES = {
    "cmath": "cmath_zh_tw.jsonl",
    "gaokao_mathcloze": "gaokao-mathcloze_zh_tw.jsonl",
    "gaokao_mathqa": "gaokao-mathqa_zh_tw.jsonl",
    "gsm8k": "gsm8k_zh_tw.jsonl",
    "math": "math_zh_tw.jsonl",
    "minif2f": "minif2f_zh_tw.jsonl",
    "mmlu_stem": "mmlu_stem_zh_tw.jsonl",
    "ocw": "ocw_zh_tw.jsonl",
}

TRAIN_ONLY = {
    "cmath",
    "gaokao_mathcloze",
    "gaokao_mathqa",
    "minif2f",
}

EVAL_ONLY = {"gsm8k", "ocw"}

# These records have lost the actual function/diagram/question context. Their
# surviving text cannot determine the reference answer, so they must not be
# used for either training or evaluation.
KNOWN_QUARANTINE_IDS = {
    "math_002668",
    "math_003272",
    "math_003311",
    "math_001044",
    "math_001825",
}


@dataclass(frozen=True)
class BuildArgs:
    raw_dir: Path
    output_dir: Path
    seed: int
    math_train_size: int
    math_eval_size: int
    mmlu_eval_size: int
    force: bool


class BuildError(RuntimeError):
    """Raised for input or configuration errors that should stop the build."""


def normalize_text(value: Any) -> str | None:
    """Normalize text conservatively without changing mathematical meaning."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in value.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def normalize_options(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_options(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): normalize_options(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return normalize_text(value)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise BuildError(f"Cannot open input file {path}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise BuildError(
                    f"Expected an object in {path} at line {line_number}"
                )
            value["_source_line"] = line_number
            yield value


def normalize_record(raw: dict[str, Any], dataset: str) -> dict[str, Any]:
    record_id = normalize_text(raw.get("id"))
    question = normalize_text(raw.get("question"))
    if not record_id:
        raise BuildError(
            f"Missing id in {DATASET_FILES[dataset]} line {raw['_source_line']}"
        )
    if not question:
        raise BuildError(
            f"Missing question for {record_id} in {DATASET_FILES[dataset]}"
        )

    return {
        "id": record_id,
        "original_id": normalize_text(raw.get("original_id")),
        "dataset": dataset,
        "question": question,
        "options": normalize_options(raw.get("options")),
        "answer": normalize_text(raw.get("answer")),
        "reasoning": normalize_text(raw.get("reasoning")),
        "source_line": raw["_source_line"],
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_key(record: dict[str, Any]) -> str:
    return canonical_json([record["question"], record["options"]])


def deterministic_score(record: dict[str, Any], seed: int, namespace: str) -> str:
    payload = f"{seed}\0{namespace}\0{record['id']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preferred_duplicate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prefer the most complete duplicate, using id as a stable tie breaker."""
    return sorted(
        records,
        key=lambda row: (
            -(len(row.get("reasoning") or "")),
            -(len(row.get("answer") or "")),
            row["id"],
        ),
    )[0]


def clean_dataset(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return clean records, duplicate reports, and quarantined records."""
    quarantine: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    seen_ids: set[str] = set()
    for record in records:
        if record["id"] in seen_ids:
            raise BuildError(
                f"Duplicate id {record['id']!r} in dataset {record['dataset']}"
            )
        seen_ids.add(record["id"])

        if record["id"] in KNOWN_QUARANTINE_IDS:
            quarantine.append(
                {
                    **record,
                    "quarantine_reason": "known_missing_question_context",
                }
            )
        else:
            candidates.append(record)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        groups[prompt_key(record)].append(record)

    clean: list[dict[str, Any]] = []
    duplicate_reports: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        answers = sorted({row.get("answer") for row in group}, key=lambda x: x or "")
        if len(answers) > 1:
            for row in group:
                quarantine.append(
                    {
                        **row,
                        "quarantine_reason": "same_prompt_conflicting_answers",
                    }
                )
            continue

        kept = preferred_duplicate(group)
        clean.append(kept)
        if len(group) > 1:
            removed = sorted(row["id"] for row in group if row is not kept)
            duplicate_reports.append(
                {
                    "dataset": kept["dataset"],
                    "kept_id": kept["id"],
                    "removed_ids": removed,
                    "answer": kept.get("answer"),
                    "prompt_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                }
            )

    clean.sort(key=lambda row: row["id"])
    quarantine.sort(key=lambda row: (row["dataset"], row["id"]))
    duplicate_reports.sort(key=lambda row: (row["dataset"], row["kept_id"]))
    return clean, duplicate_reports, quarantine


def split_records(
    datasets: dict[str, list[dict[str, Any]]], args: BuildArgs
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    split_by_dataset: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}

    for dataset, records in datasets.items():
        if dataset in TRAIN_ONLY:
            labels = {row["id"]: "train" for row in records}
        elif dataset in EVAL_ONLY:
            labels = {row["id"]: "eval" for row in records}
        elif dataset == "math":
            required = args.math_train_size + args.math_eval_size
            if len(records) < required:
                raise BuildError(
                    f"math has {len(records)} clean rows, but {required} are required"
                )
            ranked = sorted(
                records,
                key=lambda row: (
                    deterministic_score(row, args.seed, "math"), row["id"]
                ),
            )
            train_ids = {row["id"] for row in ranked[: args.math_train_size]}
            eval_ids = {
                row["id"]
                for row in ranked[
                    args.math_train_size : args.math_train_size + args.math_eval_size
                ]
            }
            labels = {
                row["id"]: (
                    "train"
                    if row["id"] in train_ids
                    else "eval"
                    if row["id"] in eval_ids
                    else "unused"
                )
                for row in records
            }
        elif dataset == "mmlu_stem":
            if len(records) < args.mmlu_eval_size:
                raise BuildError(
                    f"mmlu_stem has {len(records)} clean rows, but "
                    f"{args.mmlu_eval_size} eval rows are required"
                )
            ranked = sorted(
                records,
                key=lambda row: (
                    deterministic_score(row, args.seed, "mmlu_stem"), row["id"]
                ),
            )
            eval_ids = {row["id"] for row in ranked[: args.mmlu_eval_size]}
            labels = {
                row["id"]: "eval" if row["id"] in eval_ids else "train"
                for row in records
            }
        else:  # pragma: no cover - guarded by the fixed configuration
            raise BuildError(f"No split rule for dataset {dataset}")

        split_rows = [{**row, "split": labels[row["id"]]} for row in records]
        split_by_dataset[dataset] = split_rows
        if dataset in {"math", "mmlu_stem"}:
            manifests[dataset] = [
                {
                    "id": row["id"],
                    "original_id": row.get("original_id"),
                    "dataset": dataset,
                    "split": labels[row["id"]],
                }
                for row in records
            ]

    return split_by_dataset, manifests


def remove_train_eval_overlap(
    split_by_dataset: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Quarantine train prompts duplicated anywhere in evaluation."""
    eval_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for dataset, rows in split_by_dataset.items():
        for row in rows:
            if row["split"] == "eval":
                eval_keys[prompt_key(row)].append((dataset, row["id"]))

    overlaps: list[dict[str, Any]] = []
    for dataset, rows in split_by_dataset.items():
        for row in rows:
            if row["split"] != "train":
                continue
            key = prompt_key(row)
            if key not in eval_keys:
                continue
            row["split"] = "excluded_overlap"
            overlaps.append(
                {
                    "train_dataset": dataset,
                    "train_id": row["id"],
                    "matching_eval_records": [
                        {"dataset": eval_dataset, "id": eval_id}
                        for eval_dataset, eval_id in sorted(eval_keys[key])
                    ],
                    "prompt_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                }
            )

    overlaps.sort(key=lambda row: (row["train_dataset"], row["train_id"]))
    return overlaps


def prompt_text(record: dict[str, Any]) -> str:
    question = record["question"]
    options = record.get("options")
    if options is None or options == "" or options == [] or options == {}:
        return question
    if isinstance(options, str):
        rendered_options = options
    elif isinstance(options, list):
        pieces = []
        for index, option in enumerate(options):
            if isinstance(option, dict) and "label" in option and "text" in option:
                pieces.append(f"({option['label']}) {option['text']}")
            else:
                pieces.append(str(option))
        rendered_options = "\n".join(pieces)
    else:
        rendered_options = "\n".join(
            f"({key}) {value}" for key, value in options.items()
        )
    return f"{question}\n\n選項：\n{rendered_options}"


def training_prompt(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "conversations": [{"role": "user", "content": prompt_text(record)}],
    }


def evaluation_prompt(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": record["id"],
        "dataset": record["dataset"],
        "turns": [prompt_text(record)],
    }


def evaluation_reference(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "question": record["question"],
        "options": record.get("options"),
        "reference_answer": record.get("answer"),
        "reference_reasoning": record.get("reasoning"),
    }


def json_ready_normalized(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "source_line"}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def output_paths(output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for dataset in DATASET_FILES:
        paths.extend(
            [
                output_dir / "normalized" / f"{dataset}.jsonl",
                output_dir / "prompts" / "train" / f"{dataset}.jsonl",
                output_dir / "prompts" / "eval" / f"{dataset}.jsonl",
                output_dir / "references" / f"{dataset}.jsonl",
            ]
        )
    paths.extend(
        [
            output_dir / "splits" / "math_split.jsonl",
            output_dir / "splits" / "mmlu_stem_split.jsonl",
            output_dir / "prompts" / "train" / "all_train.jsonl",
            output_dir / "prompts" / "eval" / "all_eval.jsonl",
            output_dir / "references" / "all_eval.jsonl",
            output_dir / "reports" / "duplicates.jsonl",
            output_dir / "reports" / "quarantine.jsonl",
            output_dir / "reports" / "train_eval_overlap.jsonl",
            output_dir / "reports" / "build_report.json",
        ]
    )
    return paths


def ensure_outputs_are_writable(args: BuildArgs) -> None:
    existing = [path for path in output_paths(args.output_dir) if path.exists()]
    if existing and not args.force:
        preview = "\n  ".join(str(path) for path in existing[:5])
        suffix = "\n  ..." if len(existing) > 5 else ""
        raise BuildError(
            "Refusing to overwrite existing generated files. Re-run with --force "
            f"after reviewing them:\n  {preview}{suffix}"
        )


def build(args: BuildArgs) -> dict[str, Any]:
    ensure_outputs_are_writable(args)

    clean_datasets: dict[str, list[dict[str, Any]]] = {}
    all_duplicates: list[dict[str, Any]] = []
    all_quarantine: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}

    for dataset, filename in DATASET_FILES.items():
        path = args.raw_dir / filename
        if not path.is_file():
            raise BuildError(f"Missing required input file: {path}")
        normalized = [normalize_record(row, dataset) for row in read_jsonl(path)]
        raw_counts[dataset] = len(normalized)
        clean, duplicates, quarantine = clean_dataset(normalized)
        clean_datasets[dataset] = clean
        all_duplicates.extend(duplicates)
        all_quarantine.extend(quarantine)

    split_by_dataset, manifests = split_records(clean_datasets, args)
    overlap_reports = remove_train_eval_overlap(split_by_dataset)

    # Split manifests must reflect any train rows excluded due to eval overlap.
    final_split_lookup = {
        (dataset, row["id"]): row["split"]
        for dataset, rows in split_by_dataset.items()
        for row in rows
    }
    for dataset, rows in manifests.items():
        for row in rows:
            row["split"] = final_split_lookup[(dataset, row["id"])]

    all_train: list[dict[str, Any]] = []
    all_eval: list[dict[str, Any]] = []
    all_references: list[dict[str, Any]] = []
    dataset_reports: dict[str, Any] = {}

    for dataset in DATASET_FILES:
        rows = split_by_dataset[dataset]
        normalized_rows = [json_ready_normalized(row) for row in rows]
        train_rows = [row for row in rows if row["split"] == "train"]
        eval_rows = [row for row in rows if row["split"] == "eval"]

        train_prompts = [training_prompt(row) for row in train_rows]
        eval_prompts = [evaluation_prompt(row) for row in eval_rows]
        references = [evaluation_reference(row) for row in eval_rows]

        write_jsonl(
            args.output_dir / "normalized" / f"{dataset}.jsonl",
            normalized_rows,
        )
        write_jsonl(
            args.output_dir / "prompts" / "train" / f"{dataset}.jsonl",
            train_prompts,
        )
        write_jsonl(
            args.output_dir / "prompts" / "eval" / f"{dataset}.jsonl",
            eval_prompts,
        )
        write_jsonl(
            args.output_dir / "references" / f"{dataset}.jsonl", references
        )

        all_train.extend(train_prompts)
        all_eval.extend(eval_prompts)
        all_references.extend(references)
        split_counts = Counter(row["split"] for row in rows)
        dataset_reports[dataset] = {
            "raw_rows": raw_counts[dataset],
            "clean_unique_rows": len(clean_datasets[dataset]),
            "duplicate_rows_removed": sum(
                len(report["removed_ids"])
                for report in all_duplicates
                if report["dataset"] == dataset
            ),
            "quarantined_rows": sum(
                1 for row in all_quarantine if row["dataset"] == dataset
            ),
            "split_counts": dict(sorted(split_counts.items())),
        }

    for dataset, rows in manifests.items():
        write_jsonl(
            args.output_dir / "splits" / f"{dataset}_split.jsonl", rows
        )

    all_train.sort(key=lambda row: (row["dataset"], row["id"]))
    all_eval.sort(key=lambda row: (row["dataset"], row["question_id"]))
    all_references.sort(key=lambda row: (row["dataset"], row["id"]))
    all_duplicates.sort(key=lambda row: (row["dataset"], row["kept_id"]))
    all_quarantine.sort(key=lambda row: (row["dataset"], row["id"]))

    write_jsonl(
        args.output_dir / "prompts" / "train" / "all_train.jsonl", all_train
    )
    write_jsonl(
        args.output_dir / "prompts" / "eval" / "all_eval.jsonl", all_eval
    )
    write_jsonl(
        args.output_dir / "references" / "all_eval.jsonl", all_references
    )
    write_jsonl(
        args.output_dir / "reports" / "duplicates.jsonl", all_duplicates
    )
    write_jsonl(
        args.output_dir / "reports" / "quarantine.jsonl", all_quarantine
    )
    write_jsonl(
        args.output_dir / "reports" / "train_eval_overlap.jsonl",
        overlap_reports,
    )

    report = {
        "build_parameters": {
            "seed": args.seed,
            "math_train_size": args.math_train_size,
            "math_eval_size": args.math_eval_size,
            "mmlu_eval_size": args.mmlu_eval_size,
            "split_algorithm": "sha256(seed, dataset, id)",
        },
        "datasets": dataset_reports,
        "totals": {
            "raw_rows": sum(raw_counts.values()),
            "clean_unique_rows": sum(len(rows) for rows in clean_datasets.values()),
            "train_prompts": len(all_train),
            "eval_prompts": len(all_eval),
            "duplicate_rows_removed": sum(
                len(report["removed_ids"]) for report in all_duplicates
            ),
            "quarantined_rows": len(all_quarantine),
            "train_rows_excluded_for_eval_overlap": len(overlap_reports),
        },
        "output_files": {
            "combined_train_prompts": "prompts/train/all_train.jsonl",
            "combined_eval_prompts": "prompts/eval/all_eval.jsonl",
            "combined_eval_references": "references/all_eval.jsonl",
            "quarantine": "reports/quarantine.jsonl",
            "duplicates": "reports/duplicates.jsonl",
            "train_eval_overlap": "reports/train_eval_overlap.jsonl",
        },
    }
    write_json(args.output_dir / "reports" / "build_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> BuildArgs:
    parser = argparse.ArgumentParser(
        description="Clean, split, and export private DeepSpec datasets."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--math-train-size", type=int, default=2000)
    parser.add_argument("--math-eval-size", type=int, default=500)
    parser.add_argument("--mmlu-eval-size", type=int, default=318)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files previously generated by this script.",
    )
    parsed = parser.parse_args(argv)
    for name in ("math_train_size", "math_eval_size", "mmlu_eval_size"):
        if getattr(parsed, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return BuildArgs(
        raw_dir=parsed.raw_dir.resolve(),
        output_dir=parsed.output_dir.resolve(),
        seed=parsed.seed,
        math_train_size=parsed.math_train_size,
        math_eval_size=parsed.math_eval_size,
        mmlu_eval_size=parsed.mmlu_eval_size,
        force=parsed.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build(args)
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nBuild completed: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
