#!/usr/bin/env python3
"""Download, clean, split, and merge the Hugging Face DeepSpec datasets.

This is stage 2 of the private dataset build.  It reads the stage-1 outputs
created by build_private_datasets.py, downloads five pinned Hugging Face
datasets, and writes a new combined dataset without modifying stage 1.

The split is deterministic.  Evaluation has priority over training, and an
exact normalized final-user prompt is never allowed in both train and eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SOURCE_SPECS = {
    "tw_math_reasoning_2k": {
        "repo": "twinkle-ai/tw-math-reasoning-2k",
        "revision": "ef93a3601ac51c3d863eb82f2d6c6657ec930051",
        "split": "train",
    },
    "tw_reasoning_instruct_50k": {
        "repo": "twinkle-ai/tw-reasoning-instruct-50k",
        "revision": "f5e1f079b9d2b769f82cd370e292ffdcea1c4f2c",
        "split": "train",
    },
    "tw_function_call_reasoning_10k": {
        "repo": "twinkle-ai/tw-function-call-reasoning-10k",
        "revision": "53910fd73cb6695a0990ee8c79cc29ea6f3b0a84",
        "split": "train",
    },
    "tw_instruct_500k": {
        "repo": "lianghsun/tw-instruct-500k",
        "revision": "6ccf638af64dc78fb840c970f2ea9509948f1378",
        "split": "train",
    },
    "bailong_bench": {
        "repo": "INX-TEXT/Bailong-bench",
        "revision": "c2b3a333691e7ec8a9fb1c0e645f8f6a86417b97",
        "filename": "Bailong-bench.json",
        "split": "eval",
    },
}

DATASET_ORDER = tuple(SOURCE_SPECS)


@dataclass(frozen=True)
class BuildArgs:
    base_data_dir: Path
    output_dir: Path
    seed: int
    reasoning_train_size: int
    function_train_size: int
    function_eval_size: int
    instruct_train_size: int
    instruct_eval_size: int
    env_file: Path
    cache_dir: Path | None
    force: bool


class BuildError(RuntimeError):
    """A user-correctable input, access, or data-quality error."""


def normalize_text(value: Any) -> str | None:
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise BuildError(f"Cannot open {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise BuildError(f"Expected object in {path}:{line_number}")
            yield row


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


def load_env_file(path: Path) -> None:
    """Load only HF_TOKEN from a simple .env file, without printing it."""
    if os.environ.get("HF_TOKEN") or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "HF_TOKEN":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ["HF_TOKEN"] = value
        return


def load_hf_sources(args: BuildArgs) -> dict[str, list[dict[str, Any]]]:
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise BuildError(
            "Missing dependency 'datasets'. Install it in the active environment "
            "with: pip install 'datasets>=3,<5'"
        ) from exc

    load_env_file(args.env_file)
    token = os.environ.get("HF_TOKEN") or None
    sources: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASET_ORDER:
        spec = SOURCE_SPECS[dataset]
        print(f"Loading {spec['repo']} at {spec['revision']} ...", file=sys.stderr)
        try:
            if dataset == "bailong_bench":
                # The upstream file is JSONL despite its .json suffix.  Its
                # Prompt column mixes strings and arrays, so datasets' Arrow
                # schema inference cannot load it directly.
                local_path = hf_hub_download(
                    repo_id=spec["repo"],
                    repo_type="dataset",
                    filename=spec["filename"],
                    revision=spec["revision"],
                    token=token,
                    cache_dir=str(args.cache_dir) if args.cache_dir else None,
                )
                loaded = list(read_jsonl(Path(local_path)))
            else:
                loaded = load_dataset(
                    spec["repo"],
                    split=spec["split"],
                    revision=spec["revision"],
                    token=token,
                    cache_dir=str(args.cache_dir) if args.cache_dir else None,
                )
        except Exception as exc:
            raise BuildError(f"Failed to load {spec['repo']}: {exc}") from exc
        sources[dataset] = [dict(row) for row in loaded]
    return sources


def normalize_messages(value: Any, dataset: str, source_index: int) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise BuildError(f"{dataset} row {source_index} has no message list")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = normalize_text(item.get("role") if "role" in item else item.get("from"))
        content = normalize_text(
            item.get("content") if "content" in item else item.get("value")
        )
        role_map = {"human": "user", "gpt": "assistant"}
        role = role_map.get(role or "", role)
        if role not in {"system", "user", "assistant", "tool"} or not content:
            continue
        result.append({"role": role, "content": content})
    if not result:
        raise BuildError(f"{dataset} row {source_index} has no usable messages")
    return result


def prompt_before_final_assistant(
    messages: list[dict[str, str]], dataset: str, source_index: int
) -> tuple[list[dict[str, str]], str]:
    assistant_indices = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_indices:
        raise BuildError(f"{dataset} row {source_index} has no assistant reference")
    final_index = assistant_indices[-1]
    prefix = messages[:final_index]
    answer = messages[final_index]["content"]
    if not any(message["role"] == "user" for message in prefix):
        raise BuildError(f"{dataset} row {source_index} has no user prompt")
    return prefix, answer


def stable_id(dataset: str, source_id: Any, prompt: str) -> str:
    if source_id is not None and normalize_text(source_id):
        suffix = normalize_text(source_id)
    else:
        suffix = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    safe_suffix = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in suffix)
    return f"{dataset}_{safe_suffix}"


def final_user_text(messages: Sequence[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    raise BuildError("Conversation has no user message")


def prompt_key_from_text(text: str) -> str:
    normalized = normalize_text(text)
    return normalized or ""


def normalize_hf_row(dataset: str, row: dict[str, Any], source_index: int) -> dict[str, Any]:
    if dataset == "tw_math_reasoning_2k":
        question = normalize_text(row.get("problem_zhtw"))
        if not question:
            raise BuildError(f"{dataset} row {source_index} lacks problem_zhtw")
        messages = [{"role": "user", "content": question}]
        answer = normalize_text(row.get("answer"))
        reasoning = normalize_text(row.get("think"))
        source_id = hashlib.sha256(
            (normalize_text(row.get("problem")) or question).encode("utf-8")
        ).hexdigest()[:16]
    elif dataset == "tw_reasoning_instruct_50k":
        question = normalize_text(row.get("input"))
        if not question:
            raise BuildError(f"{dataset} row {source_index} lacks input")
        messages = [{"role": "user", "content": question}]
        answer = normalize_text(row.get("output"))
        reasoning = normalize_text(row.get("think"))
        source_id = None
    elif dataset == "tw_function_call_reasoning_10k":
        all_messages = normalize_messages(row.get("messages"), dataset, source_index)
        messages, message_answer = prompt_before_final_assistant(
            all_messages, dataset, source_index
        )
        question = final_user_text(messages)
        answer = normalize_text(row.get("answer")) or message_answer
        reasoning = normalize_text(row.get("think"))
        source_id = row.get("id")
    elif dataset == "tw_instruct_500k":
        all_messages = normalize_messages(row.get("conversations"), dataset, source_index)
        messages, answer = prompt_before_final_assistant(
            all_messages, dataset, source_index
        )
        question = final_user_text(messages)
        reasoning = None
        source_id = row.get("id")
        eval_turns = None
        category = None
    elif dataset == "bailong_bench":
        raw_prompt = row.get("Prompt")
        if isinstance(raw_prompt, str):
            eval_turns = [normalize_text(raw_prompt)]
        elif isinstance(raw_prompt, list):
            eval_turns = [normalize_text(turn) for turn in raw_prompt]
        else:
            eval_turns = []
        if not eval_turns or any(not turn for turn in eval_turns):
            raise BuildError(f"{dataset} row {source_index} has an invalid Prompt")
        category = normalize_text(row.get("Type"))
        if not category:
            raise BuildError(f"{dataset} row {source_index} lacks Type")
        question = eval_turns[-1]
        messages = [{"role": "user", "content": eval_turns[0]}]
        answer = None
        reasoning = None
        source_id = f"{source_index:06d}"
    else:
        raise BuildError(f"Unknown dataset adapter: {dataset}")

    if dataset != "bailong_bench" and not answer:
        raise BuildError(f"{dataset} row {source_index} has an empty reference answer")
    if dataset != "bailong_bench":
        eval_turns = None
        category = None
    prompt_keys = [prompt_key_from_text(turn) for turn in (eval_turns or [question])]
    canonical_prompt = canonical_json(prompt_keys)
    record_id = stable_id(dataset, source_id, canonical_prompt)
    return {
        "id": record_id,
        "dataset": dataset,
        "source_index": source_index,
        "source_revision": SOURCE_SPECS[dataset]["revision"],
        "conversations": messages,
        "question": question,
        "reference_answer": answer,
        "reference_reasoning": reasoning,
        "eval_turns": eval_turns,
        "category": category,
        "prompt_key": canonical_prompt if len(prompt_keys) > 1 else prompt_keys[0],
        "prompt_keys": prompt_keys,
    }


def preferred_duplicate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            -len(row.get("reference_answer") or ""),
            -len(row.get("reference_reasoning") or ""),
            row["id"],
        ),
    )[0]


def clean_hf_dataset(
    rows: Sequence[dict[str, Any]], dataset: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        try:
            normalized.append(normalize_hf_row(dataset, raw, index))
        except BuildError as exc:
            quarantine.append(
                {
                    "dataset": dataset,
                    "source_index": index,
                    "quarantine_reason": "invalid_schema_or_empty_content",
                    "detail": str(exc),
                }
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        groups[row["prompt_key"]].append(row)

    clean: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        kept = preferred_duplicate(group)
        clean.append(kept)
        if len(group) > 1:
            duplicates.append(
                {
                    "dataset": dataset,
                    "kept_id": kept["id"],
                    "removed_ids": sorted(row["id"] for row in group if row is not kept),
                    "prompt_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    "conflicting_reference_answers": len(
                        {row["reference_answer"] for row in group}
                    )
                    > 1,
                }
            )
    clean.sort(key=lambda row: row["id"])
    return clean, duplicates, quarantine


def deterministic_rank(row: dict[str, Any], seed: int, namespace: str) -> tuple[str, str]:
    payload = f"{seed}\0{namespace}\0{row['id']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), row["id"]


def render_turn(messages: Sequence[dict[str, str]]) -> str:
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    labels = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
    return "\n\n".join(
        f"[{labels[message['role']]}]\n{message['content']}" for message in messages
    )


def training_prompt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "conversations": row["conversations"],
    }


def evaluation_prompt(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": row["id"],
        "dataset": row["dataset"],
        "turns": row.get("eval_turns") or [render_turn(row["conversations"])],
    }


def evaluation_reference(row: dict[str, Any]) -> dict[str, Any]:
    reference = {
        "id": row["id"],
        "dataset": row["dataset"],
        "question": row["question"],
        "reference_answer": row["reference_answer"],
        "reference_reasoning": row["reference_reasoning"],
    }
    if row.get("category") is not None:
        reference["category"] = row["category"]
        reference["scoring"] = "llm_as_a_judge"
    return reference


def base_train_key(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise BuildError("Stage-1 training prompt lacks conversations")
    messages = normalize_messages(conversations, "stage1", -1)
    return prompt_key_from_text(final_user_text(messages))


def base_eval_keys(row: dict[str, Any]) -> set[str]:
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns or not isinstance(turns[-1], str):
        raise BuildError("Stage-1 evaluation prompt lacks turns")
    return {prompt_key_from_text(turn) for turn in turns}


def row_prompt_keys(row: dict[str, Any]) -> set[str]:
    keys = row.get("prompt_keys")
    if isinstance(keys, list) and keys:
        return set(keys)
    return {row["prompt_key"]}


def select_first_eligible(
    ranked: Sequence[dict[str, Any]],
    size: int,
    forbidden_keys: set[str],
    selected_keys: set[str],
    label: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in ranked:
        keys = row_prompt_keys(row)
        if keys & forbidden_keys or keys & selected_keys:
            continue
        selected.append(row)
        selected_keys.update(keys)
        if len(selected) == size:
            break
    if len(selected) != size:
        raise BuildError(
            f"Could select only {len(selected)} of {size} required rows for {label}"
        )
    return selected


def ensure_output_is_writable(args: BuildArgs) -> None:
    marker = args.output_dir / "reports" / "build_report.json"
    if marker.exists() and not args.force:
        raise BuildError(
            f"Refusing to overwrite {args.output_dir}. Review it, then re-run with --force."
        )


def build_from_sources(
    args: BuildArgs, raw_sources: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    ensure_output_is_writable(args)
    base_train_path = args.base_data_dir / "prompts" / "train" / "all_train.jsonl"
    base_eval_path = args.base_data_dir / "prompts" / "eval" / "all_eval.jsonl"
    base_ref_path = args.base_data_dir / "references" / "all_eval.jsonl"
    for path in (base_train_path, base_eval_path, base_ref_path):
        if not path.is_file():
            raise BuildError(f"Missing stage-1 output: {path}")

    base_train = list(read_jsonl(base_train_path))
    base_eval = list(read_jsonl(base_eval_path))
    base_refs = list(read_jsonl(base_ref_path))
    if len(base_eval) != len(base_refs):
        raise BuildError("Stage-1 eval prompt/reference counts differ")

    clean: dict[str, list[dict[str, Any]]] = {}
    all_duplicates: list[dict[str, Any]] = []
    all_quarantine: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}
    for dataset in DATASET_ORDER:
        if dataset not in raw_sources:
            raise BuildError(f"Missing loaded source {dataset}")
        raw_counts[dataset] = len(raw_sources[dataset])
        rows, duplicates, quarantine = clean_hf_dataset(raw_sources[dataset], dataset)
        clean[dataset] = rows
        all_duplicates.extend(duplicates)
        all_quarantine.extend(quarantine)

    bailong_turn_counts = Counter(
        len(row.get("eval_turns") or []) for row in clean["bailong_bench"]
    )
    if len(clean["bailong_bench"]) != 140 or bailong_turn_counts != Counter({1: 130, 3: 10}):
        raise BuildError(
            "Pinned Bailong Bench must contain 140 clean rows "
            "(130 single-turn and 10 three-turn); got "
            f"{len(clean['bailong_bench'])} rows with turn counts "
            f"{dict(sorted(bailong_turn_counts.items()))}"
        )

    existing_eval_keys: set[str] = set()
    for row in base_eval:
        existing_eval_keys.update(base_eval_keys(row))
    hf_eval_keys: set[str] = set()
    hf_eval: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASET_ORDER}

    # Evaluation is chosen first and never duplicates an existing evaluation item.
    # Bailong Bench is a fixed 140-item benchmark and must be added in full.
    bailong_ranked = sorted(clean["bailong_bench"], key=lambda row: row["id"])
    hf_eval["bailong_bench"] = bailong_ranked
    for row in bailong_ranked:
        hf_eval_keys.update(row_prompt_keys(row))

    for dataset, size in (
        ("tw_function_call_reasoning_10k", args.function_eval_size),
        ("tw_instruct_500k", args.instruct_eval_size),
    ):
        ranked = sorted(clean[dataset], key=lambda row: deterministic_rank(row, args.seed, f"{dataset}:eval"))
        hf_eval[dataset] = select_first_eligible(
            ranked, size, existing_eval_keys, hf_eval_keys, f"{dataset} eval"
        )

    all_eval_keys = existing_eval_keys | hf_eval_keys

    # New evaluation wins over an old stage-1 training duplicate.
    filtered_base_train: list[dict[str, Any]] = []
    excluded_base_train: list[dict[str, Any]] = []
    train_keys: set[str] = set()
    for row in base_train:
        key = base_train_key(row)
        if key in all_eval_keys:
            excluded_base_train.append(
                {
                    "dataset": row.get("dataset"),
                    "id": row.get("id"),
                    "reason": "matches_combined_evaluation",
                    "prompt_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                }
            )
            continue
        if key in train_keys:
            continue
        filtered_base_train.append(row)
        train_keys.add(key)

    hf_train: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASET_ORDER}
    train_exclusions: list[dict[str, Any]] = []

    def choose_train(dataset: str, requested: int | None) -> None:
        ranked = sorted(clean[dataset], key=lambda row: deterministic_rank(row, args.seed, f"{dataset}:train"))
        eval_ids = {row["id"] for row in hf_eval[dataset]}
        candidates = [row for row in ranked if row["id"] not in eval_ids]
        if requested is None:
            selected = []
            for row in candidates:
                keys = row_prompt_keys(row)
                if keys & all_eval_keys or keys & train_keys:
                    train_exclusions.append(
                        {
                            "dataset": dataset,
                            "id": row["id"],
                            "reason": "eval_overlap" if keys & all_eval_keys else "train_duplicate",
                            "prompt_key_sha256": hashlib.sha256(row["prompt_key"].encode("utf-8")).hexdigest(),
                        }
                    )
                    continue
                selected.append(row)
                train_keys.update(keys)
        else:
            selected = select_first_eligible(
                candidates, requested, all_eval_keys, train_keys, f"{dataset} train"
            )
        hf_train[dataset] = selected

    choose_train("tw_math_reasoning_2k", None)
    choose_train("tw_reasoning_instruct_50k", args.reasoning_train_size)
    choose_train("tw_function_call_reasoning_10k", args.function_train_size)
    choose_train("tw_instruct_500k", args.instruct_train_size)
    # Bailong Bench is evaluation-only.

    split_lookup: dict[tuple[str, str], str] = {}
    for dataset in DATASET_ORDER:
        for row in clean[dataset]:
            split_lookup[(dataset, row["id"])] = "unused"
        for row in hf_eval[dataset]:
            split_lookup[(dataset, row["id"])] = "eval"
        for row in hf_train[dataset]:
            split_lookup[(dataset, row["id"])] = "train"
    for row in train_exclusions:
        split_lookup[(row["dataset"], row["id"])] = f"excluded_{row['reason']}"

    combined_train = filtered_base_train + [
        training_prompt(row) for dataset in DATASET_ORDER for row in hf_train[dataset]
    ]
    combined_eval = base_eval + [
        evaluation_prompt(row) for dataset in DATASET_ORDER for row in hf_eval[dataset]
    ]
    combined_refs = base_refs + [
        evaluation_reference(row) for dataset in DATASET_ORDER for row in hf_eval[dataset]
    ]
    combined_train.sort(key=lambda row: (row.get("dataset", ""), row.get("id", "")))
    combined_eval.sort(key=lambda row: (row.get("dataset", ""), row.get("question_id", "")))
    combined_refs.sort(key=lambda row: (row.get("dataset", ""), row.get("id", "")))

    dataset_reports: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        split_counts = Counter(
            split_lookup[(dataset, row["id"])] for row in clean[dataset]
        )
        dataset_reports[dataset] = {
            "repo": SOURCE_SPECS[dataset]["repo"],
            "revision": SOURCE_SPECS[dataset]["revision"],
            "raw_rows": raw_counts[dataset],
            "clean_unique_rows": len(clean[dataset]),
            "duplicate_rows_removed": sum(
                len(row["removed_ids"]) for row in all_duplicates if row["dataset"] == dataset
            ),
            "quarantined_rows": sum(1 for row in all_quarantine if row["dataset"] == dataset),
            "split_counts": dict(sorted(split_counts.items())),
        }
        if dataset == "bailong_bench":
            dataset_reports[dataset]["turn_counts"] = {
                str(turn_count): count
                for turn_count, count in sorted(bailong_turn_counts.items())
            }
            dataset_reports[dataset]["category_counts"] = dict(
                sorted(Counter(row["category"] for row in clean[dataset]).items())
            )
            dataset_reports[dataset]["reference_answers_available"] = False
            dataset_reports[dataset]["scoring"] = "llm_as_a_judge"

        normalized_rows = []
        for row in clean[dataset]:
            normalized_rows.append(
                {
                    key: value
                    for key, value in {**row, "split": split_lookup[(dataset, row["id"])]}.items()
                    if key not in {"prompt_key", "prompt_keys"}
                }
            )
        write_jsonl(args.output_dir / "normalized" / f"{dataset}.jsonl", normalized_rows)
        write_jsonl(
            args.output_dir / "splits" / f"{dataset}_split.jsonl",
            [
                {
                    "id": row["id"],
                    "dataset": dataset,
                    "source_index": row["source_index"],
                    "source_revision": row["source_revision"],
                    "split": split_lookup[(dataset, row["id"])],
                }
                for row in clean[dataset]
            ],
        )
        write_jsonl(
            args.output_dir / "prompts" / "train" / f"{dataset}.jsonl",
            [training_prompt(row) for row in hf_train[dataset]],
        )
        write_jsonl(
            args.output_dir / "prompts" / "eval" / f"{dataset}.jsonl",
            [evaluation_prompt(row) for row in hf_eval[dataset]],
        )
        write_jsonl(
            args.output_dir / "references" / f"{dataset}.jsonl",
            [evaluation_reference(row) for row in hf_eval[dataset]],
        )

    write_jsonl(args.output_dir / "prompts" / "train" / "all_train.jsonl", combined_train)
    write_jsonl(args.output_dir / "prompts" / "eval" / "all_eval.jsonl", combined_eval)
    write_jsonl(args.output_dir / "references" / "all_eval.jsonl", combined_refs)
    write_jsonl(args.output_dir / "reports" / "duplicates.jsonl", all_duplicates)
    write_jsonl(args.output_dir / "reports" / "quarantine.jsonl", all_quarantine)
    write_jsonl(args.output_dir / "reports" / "hf_train_exclusions.jsonl", train_exclusions)
    write_jsonl(args.output_dir / "reports" / "base_train_excluded_for_new_eval.jsonl", excluded_base_train)

    report = {
        "build_parameters": {
            "seed": args.seed,
            "split_algorithm": "sha256(seed, dataset:split, stable_id)",
            "reasoning_train_size": args.reasoning_train_size,
            "function_train_size": args.function_train_size,
            "function_eval_size": args.function_eval_size,
            "instruct_train_size": args.instruct_train_size,
            "instruct_eval_size": args.instruct_eval_size,
        },
        "stage1": {
            "input_directory": str(args.base_data_dir),
            "train_prompts_before_merge": len(base_train),
            "eval_prompts_before_merge": len(base_eval),
            "train_removed_for_new_eval": len(excluded_base_train),
        },
        "datasets": dataset_reports,
        "totals": {
            "hf_raw_rows": sum(raw_counts.values()),
            "hf_clean_unique_rows": sum(len(rows) for rows in clean.values()),
            "combined_train_prompts": len(combined_train),
            "combined_eval_prompts": len(combined_eval),
            "combined_eval_references": len(combined_refs),
            "hf_duplicate_rows_removed": sum(len(row["removed_ids"]) for row in all_duplicates),
            "hf_quarantined_rows": len(all_quarantine),
            "hf_train_exclusions": len(train_exclusions),
        },
    }
    write_json(args.output_dir / "reports" / "build_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> BuildArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-train-size", type=int, default=5500)
    parser.add_argument("--function-train-size", type=int, default=1000)
    parser.add_argument("--function-eval-size", type=int, default=200)
    parser.add_argument("--instruct-train-size", type=int, default=7000)
    parser.add_argument("--instruct-eval-size", type=int, default=500)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parsed = parser.parse_args(argv)
    for name in (
        "reasoning_train_size",
        "function_train_size",
        "function_eval_size",
        "instruct_train_size",
        "instruct_eval_size",
    ):
        if getattr(parsed, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return BuildArgs(
        base_data_dir=parsed.base_data_dir.resolve(),
        output_dir=parsed.output_dir.resolve(),
        seed=parsed.seed,
        reasoning_train_size=parsed.reasoning_train_size,
        function_train_size=parsed.function_train_size,
        function_eval_size=parsed.function_eval_size,
        instruct_train_size=parsed.instruct_train_size,
        instruct_eval_size=parsed.instruct_eval_size,
        env_file=parsed.env_file.resolve(),
        cache_dir=parsed.cache_dir.resolve() if parsed.cache_dir else None,
        force=parsed.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sources = load_hf_sources(args)
        report = build_from_sources(args, sources)
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nBuild completed: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
