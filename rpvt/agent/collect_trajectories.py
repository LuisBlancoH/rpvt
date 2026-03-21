"""Collect agent trajectories for DPO training.

Runs the agent on GAIA tasks multiple times with temperature sampling.
Saves successful and failed trajectories for preference learning.

Usage:
    python -m rpvt.agent.collect_trajectories --max-tasks 20 --runs-per-task 5
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from rpvt.agent.eval_gaia_rlm import check_answer, normalize_answer
from rpvt.agent.file_readers import read_file
from rpvt.agent.rlm_agent import (
    FOLLOWUP_PROMPT,
    SYSTEM_PROMPT,
    _extract_answer,
    execute_python,
    extract_code,
)

GAIA_REPO = "gaia-benchmark/GAIA"


def collect_trajectory(agent, question, file_content=None, file_path=None,
                       temperature=0.7):
    """Run agent and record the full trajectory (messages + outputs).

    Returns (answer, trajectory_dict).
    """
    context_info = ""
    if file_content:
        context_info = (
            f"\nDocument loaded as `context` ({len(file_content)} chars)."
            f"\nFile path: {file_path}"
        )

    task_prompt = f"{question}{context_info}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    # Record every step
    steps = []
    final_answer = None

    for step in range(agent.max_steps):
        response = agent._generate(
            messages, max_new_tokens=400, temperature=temperature
        )

        # Check for direct ANSWER:
        answer = _extract_answer(response)
        if answer is not None:
            steps.append({"role": "assistant", "content": response, "type": "answer"})
            final_answer = answer
            break

        code = extract_code(response)
        if code:
            output = execute_python(
                code, context=file_content, file_path=file_path,
            )
            print(f"    [step {step+1}] → {output[:200]}")

            steps.append({
                "role": "assistant",
                "content": response,
                "type": "code",
                "code_output": output,
            })

            # Check code output for ANSWER:
            answer = _extract_answer(output)
            if answer is not None:
                final_answer = answer
                break

            # Last step — take output as answer
            if step == agent.max_steps - 1:
                if output and not output.startswith("ERROR:"):
                    lines = output.strip().split("\n")
                    final_answer = lines[-1].strip()
                else:
                    final_answer = "UNKNOWN"
                break

            # Build followup
            messages.append({"role": "assistant", "content": response})
            remaining = agent.max_steps - step - 1
            if output.startswith("ERROR:"):
                instruction = f"Fix the error. {remaining} step(s) left."
            elif remaining == 1:
                instruction = "Last step. Give your ANSWER:"
            else:
                instruction = f"{remaining} step(s) left."

            followup = FOLLOWUP_PROMPT.format(
                output=output, instruction=instruction
            )
            messages.append({"role": "user", "content": followup})
            steps.append({"role": "user", "content": followup, "type": "followup"})
        else:
            # No code — push to use code
            if step < agent.max_steps - 1:
                messages.append({"role": "assistant", "content": response})
                push = "Use Python code to solve this. Write a ```python block."
                messages.append({"role": "user", "content": push})
                steps.append({"role": "assistant", "content": response, "type": "text"})
                steps.append({"role": "user", "content": push, "type": "push"})
            else:
                lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
                if lines:
                    final_answer = lines[-1]
                steps.append({"role": "assistant", "content": response, "type": "text"})
                break

    if final_answer is None:
        final_answer = "UNKNOWN"

    # Reconstruct the full conversation for training
    # Start with system + user, then interleave assistant/user from steps
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    for s in steps:
        conversation.append({"role": s["role"], "content": s["content"]})

    return final_answer, {
        "messages": conversation,
        "steps": steps,
        "answer": final_answer,
    }


def download_task_file(task, cache_dir):
    file_name = task.get("file_name", "")
    if not file_name:
        return None
    local_path = cache_dir / file_name
    if local_path.exists():
        return local_path
    try:
        path = hf_hub_download(
            repo_id=GAIA_REPO,
            filename=f"2023/validation/{file_name}",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        return Path(path)
    except Exception as e:
        print(f"  Failed to download {file_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Collect agent trajectories")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--runs-per-task", type=int, default=5,
                        help="Number of attempts per task for diversity")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--file-only", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default="results/trajectories")
    args = parser.parse_args()

    print("Loading GAIA dataset...")
    ds = load_dataset(GAIA_REPO, "2023_all", split="validation")
    tasks = list(ds)

    if args.level:
        tasks = [t for t in tasks if t["Level"] == args.level]
    if args.file_only:
        tasks = [t for t in tasks if t.get("file_name")]
    # Filter out web-requiring tasks
    def needs_web(t):
        meta = t.get("Annotator Metadata", {}) or {}
        tools = meta.get("Tools", "").lower()
        return "web" in tools or "browser" in tools or "search" in tools
    tasks = [t for t in tasks if not needs_web(t)]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"  Tasks: {len(tasks)}, runs per task: {args.runs_per_task}")

    cache_dir = Path(args.output_dir) / "files"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from rpvt.agent.rlm_agent import RLMAgent
    agent = RLMAgent(
        model_name=args.model_name,
        max_steps=args.max_steps,
        quantize=args.quantize,
    )

    all_trajectories = []
    stats = {"total_runs": 0, "correct": 0, "wrong": 0, "tasks_with_good": 0}

    for i, task in enumerate(tasks):
        question = task["Question"]
        gold = task["Final answer"]
        level = task["Level"]
        task_id = task["task_id"]
        has_file = bool(task.get("file_name"))

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(tasks)}] L{level} {'(file)' if has_file else ''}")
        print(f"  Q: {question}")
        print(f"  Gold: {gold}")

        file_content = None
        file_path_str = None
        if has_file:
            file_path = download_task_file(task, cache_dir)
            if file_path:
                file_path_str = str(file_path.resolve())
                file_content, err = read_file(file_path)
                if err:
                    print(f"  File error: {err}")
                    file_content = None

        task_trajectories = {"good": [], "bad": []}

        for run in range(args.runs_per_task):
            print(f"\n  --- Run {run+1}/{args.runs_per_task} ---")
            t0 = time.time()
            try:
                predicted, trajectory = collect_trajectory(
                    agent, question,
                    file_content=file_content,
                    file_path=file_path_str,
                    temperature=args.temperature,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                predicted = "ERROR"
                trajectory = {"messages": [], "steps": [], "answer": "ERROR"}
            elapsed = time.time() - t0

            is_correct = check_answer(predicted, gold)
            status = "CORRECT" if is_correct else "WRONG"
            print(f"  Answer: {predicted[:150]}")
            print(f"  [{status}] ({elapsed:.1f}s)")

            trajectory["task_id"] = task_id
            trajectory["question"] = question
            trajectory["gold"] = gold
            trajectory["predicted"] = predicted
            trajectory["correct"] = is_correct
            trajectory["level"] = level
            trajectory["run"] = run
            trajectory["time"] = elapsed

            if is_correct:
                task_trajectories["good"].append(trajectory)
                stats["correct"] += 1
            else:
                task_trajectories["bad"].append(trajectory)
                stats["wrong"] += 1
            stats["total_runs"] += 1

        # Save DPO pairs: every (good, bad) combination for this task
        if task_trajectories["good"]:
            stats["tasks_with_good"] += 1

        all_trajectories.append({
            "task_id": task_id,
            "question": question,
            "gold": gold,
            "level": level,
            "good": task_trajectories["good"],
            "bad": task_trajectories["bad"],
        })

        # Save incrementally
        with open(output_dir / "trajectories.json", "w") as f:
            json.dump(all_trajectories, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Correct: {stats['correct']} ({100*stats['correct']/max(stats['total_runs'],1):.1f}%)")
    print(f"  Tasks with ≥1 success: {stats['tasks_with_good']}/{len(tasks)}")
    print(f"  Saved to {output_dir / 'trajectories.json'}")

    # Also save DPO-ready pairs
    dpo_pairs = []
    for t in all_trajectories:
        for good in t["good"]:
            for bad in t["bad"]:
                dpo_pairs.append({
                    "task_id": t["task_id"],
                    "question": t["question"],
                    "chosen": good["messages"],
                    "rejected": bad["messages"],
                })

    with open(output_dir / "dpo_pairs.json", "w") as f:
        json.dump(dpo_pairs, f, indent=2)
    print(f"  DPO pairs: {len(dpo_pairs)} (saved to dpo_pairs.json)")


if __name__ == "__main__":
    main()
