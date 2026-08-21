from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import (
    Gemma4DSparkEvaluator,
    Llama3DSparkEvaluator,
    Qwen3DSparkEvaluator,
)
from deepspec.eval.eagle3 import (
    Gemma4Eagle3Evaluator,
    Llama3Eagle3Evaluator,
    Qwen3Eagle3Evaluator,
)
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Llama3DSparkModel": Llama3DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Llama3Eagle3Model": Llama3Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

TASKS = [
    ("gsm8k", 50),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Evaluation dtype. auto uses BF16 when supported, otherwise FP16.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load target model, draft model, and tokenizer only from local paths.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help="Cap each selected task; useful for smoke tests.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
    "--tasks",
    nargs="+",
    choices=[name for name, _ in TASKS],
    default=None,
    help="Tasks to evaluate. Default: all tasks.",
)
    args = parser.parse_args()
    selected_tasks = args.tasks
    if selected_tasks is None:
        args.tasks = list(TASKS)
    else:
        task_limits = dict(TASKS)
        args.tasks = [
            (
                task_name,
                min(task_limits[task_name], args.max_samples_per_task)
                if args.max_samples_per_task is not None
                else task_limits[task_name],
            )
            for task_name in selected_tasks
        ]

    if args.max_samples_per_task is not None:
        if args.max_samples_per_task <= 0:
            parser.error("--max-samples-per-task must be greater than zero")
        if selected_tasks is None:
            args.tasks = [
                (task_name, min(task_limit, args.max_samples_per_task))
                for task_name, task_limit in args.tasks
            ]

    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(
        args.draft_name_or_path,
        local_files_only=args.local_files_only,
    )
    architecture = draft_config.architectures[0]
    if architecture not in EVALUATORS:
        raise ValueError(
            f"Unsupported draft architecture {architecture!r}. "
            f"Supported architectures: {sorted(EVALUATORS)}"
        )
    evaluator_cls = EVALUATORS[architecture]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
