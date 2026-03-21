"""GAIA benchmark evaluation for the RPVT agent.

Runs the agent on GAIA validation tasks and measures exact-match accuracy.
Starts with file-based tasks (our strength) and reports what's missing.

Usage:
    python -m rpvt.agent.eval_gaia
    python -m rpvt.agent.eval_gaia --level 1
    python -m rpvt.agent.eval_gaia --file-only
    python -m rpvt.agent.eval_gaia --max-tasks 10
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from rpvt.agent.core import AgentCore
from rpvt.agent.file_readers import read_file


GAIA_REPO = "gaia-benchmark/GAIA"


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    s = str(answer).strip().lower()
    # Remove trailing punctuation
    s = s.rstrip(".")
    # Normalize whitespace
    s = " ".join(s.split())
    return s


def check_answer(predicted, gold):
    """Check if predicted answer matches gold (flexible matching)."""
    pred = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact match
    if pred == gold_norm:
        return True

    # Check if gold answer is contained in prediction
    if gold_norm in pred:
        return True

    # Check if prediction starts with gold answer
    if pred.startswith(gold_norm):
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred.replace(",", ""))
        gold_num = float(gold_norm.replace(",", ""))
        if abs(pred_num - gold_num) < 0.01:
            return True
    except (ValueError, TypeError):
        pass

    return False


def download_task_file(task, cache_dir):
    """Download the attached file for a task, if any."""
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


def build_prompt(question, file_content=None):
    """Build a prompt for the agent from a GAIA task."""
    parts = []
    if file_content:
        # Truncate very long files to fit in memory
        max_chars = 8000
        if len(file_content) > max_chars:
            file_content = file_content[:max_chars] + "\n... (truncated)"
        parts.append(f"Document content:\n{file_content}\n")
    parts.append(question)
    parts.append("\nAnswer with ONLY the final answer, nothing else. Be concise.")
    return "\n".join(parts)


def run_eval(agent, tasks, cache_dir, max_tasks=None):
    """Run evaluation on a set of tasks."""
    results = []
    correct = 0
    total = 0
    skipped = 0

    if max_tasks:
        tasks = tasks[:max_tasks]

    for i, task in enumerate(tasks):
        task_id = task["task_id"][:8]
        question = task["Question"]
        gold = task["Final answer"]
        level = task["Level"]
        has_file = bool(task.get("file_name"))

        print(f"\n[{i+1}/{len(tasks)}] L{level} {'(file)' if has_file else '(no file)'}")
        print(f"  Q: {question[:120]}...")
        print(f"  Gold: {gold}")

        # Reset memory for each task
        agent.reset_memory()

        # Handle attached file
        file_content = None
        if has_file:
            file_path = download_task_file(task, cache_dir)
            if file_path:
                file_content, err = read_file(file_path)
                if err:
                    print(f"  File error: {err}")
                elif file_content:
                    # Ingest into memory
                    result = agent.ingest_text(file_content, doc_id=f"task_{task_id}")
                    if "error" not in result:
                        print(f"  Ingested: {result['n_tokens']} tokens, "
                              f"{result['n_stored']} entries")
                    else:
                        print(f"  Ingest error: {result['error']}")

        # Build prompt and generate
        prompt = build_prompt(question, file_content)
        t0 = time.time()
        try:
            predicted = agent.generate(prompt, max_new_tokens=100, use_memory=True)
        except Exception as e:
            predicted = f"ERROR: {e}"
        elapsed = time.time() - t0

        # Check answer
        is_correct = check_answer(predicted, gold)
        if is_correct:
            correct += 1
        total += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  Predicted: {predicted[:150]}")
        print(f"  [{status}] ({elapsed:.1f}s)")

        results.append({
            "task_id": task["task_id"],
            "level": level,
            "has_file": has_file,
            "question": question[:200],
            "gold": gold,
            "predicted": predicted[:200],
            "correct": is_correct,
            "time": elapsed,
        })

    return results, correct, total


def main():
    parser = argparse.ArgumentParser(description="GAIA Benchmark Evaluation")
    parser.add_argument("--model-name", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-entries", type=int, default=2048)
    parser.add_argument("--level", type=int, default=None,
                        help="Filter by level (1, 2, or 3)")
    parser.add_argument("--file-only", action="store_true",
                        help="Only run tasks with attached files")
    parser.add_argument("--no-web", action="store_true", default=True,
                        help="Skip tasks that need web browsing (default)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run all tasks (including web-dependent)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks to run")
    parser.add_argument("--output-dir", type=str,
                        default="results/gaia_eval")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # Load dataset
    print("Loading GAIA dataset...")
    ds = load_dataset(GAIA_REPO, "2023_all", split="validation")
    print(f"  {len(ds)} tasks loaded")

    # Filter tasks
    tasks = list(ds)

    if args.level:
        tasks = [t for t in tasks if t["Level"] == args.level]
        print(f"  Level {args.level}: {len(tasks)} tasks")

    if args.file_only:
        tasks = [t for t in tasks if t.get("file_name")]
        print(f"  File-only: {len(tasks)} tasks")

    if not args.all_tasks:
        # Filter out tasks that need web browsing
        def needs_web(t):
            meta = t.get("Annotator Metadata", {}) or {}
            tools = meta.get("Tools", "").lower()
            return "web" in tools or "browser" in tools or "search" in tools
        before = len(tasks)
        tasks = [t for t in tasks if not needs_web(t)]
        print(f"  No-web filter: {len(tasks)} tasks (removed {before - len(tasks)})")

    if not tasks:
        print("No tasks match the filters.")
        return

    # Setup
    cache_dir = Path(args.output_dir) / "files"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize agent
    agent = AgentCore(
        model_name=args.model_name,
        device=args.device,
        max_entries=args.max_entries,
    )

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Running GAIA evaluation: {len(tasks)} tasks")
    print(f"{'='*60}")

    results, correct, total = run_eval(
        agent, tasks, cache_dir, max_tasks=args.max_tasks
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total: {correct}/{total} correct ({100*correct/max(total,1):.1f}%)")

    # By level
    for lvl in [1, 2, 3]:
        lvl_results = [r for r in results if r["level"] == lvl]
        if lvl_results:
            lvl_correct = sum(1 for r in lvl_results if r["correct"])
            print(f"  Level {lvl}: {lvl_correct}/{len(lvl_results)} "
                  f"({100*lvl_correct/len(lvl_results):.1f}%)")

    # By file presence
    file_results = [r for r in results if r["has_file"]]
    nofile_results = [r for r in results if not r["has_file"]]
    if file_results:
        fc = sum(1 for r in file_results if r["correct"])
        print(f"  With file: {fc}/{len(file_results)} "
              f"({100*fc/len(file_results):.1f}%)")
    if nofile_results:
        nc = sum(1 for r in nofile_results if r["correct"])
        print(f"  No file: {nc}/{len(nofile_results)} "
              f"({100*nc/len(nofile_results):.1f}%)")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "summary": {
                "correct": correct,
                "total": total,
                "accuracy": correct / max(total, 1),
                "model": args.model_name,
                "max_entries": args.max_entries,
            },
            "tasks": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Show failures for analysis
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n--- Failures ({len(failures)}) ---")
        for r in failures[:10]:
            print(f"  L{r['level']} Q: {r['question'][:80]}...")
            print(f"     Gold: {r['gold']}")
            print(f"     Got:  {r['predicted'][:80]}")
            print()


if __name__ == "__main__":
    main()
