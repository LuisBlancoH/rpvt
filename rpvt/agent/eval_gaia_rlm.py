"""GAIA evaluation with RLM agent.

Usage:
    python -m rpvt.agent.eval_gaia_rlm --max-tasks 15
    python -m rpvt.agent.eval_gaia_rlm --model-name Qwen/Qwen3.5-9B --quantize
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from rpvt.agent.file_readers import read_file


GAIA_REPO = "gaia-benchmark/GAIA"


def normalize_answer(answer):
    if answer is None:
        return ""
    s = str(answer).strip().lower()
    s = s.rstrip(".")
    for prefix in ["final answer:", "answer:", "the answer is"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = s.replace("**", "")
    s = " ".join(s.split())
    s = s.replace(" ,", ",").replace(",  ", ", ").replace(",", ", ")
    s = " ".join(s.split())
    return s.strip()


def check_answer(predicted, gold):
    pred = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred == gold_norm:
        return True
    if gold_norm in pred:
        return True
    if pred.startswith(gold_norm):
        return True
    try:
        pred_num = float(pred.replace(",", ""))
        gold_num = float(gold_norm.replace(",", ""))
        if abs(pred_num - gold_num) < 0.01:
            return True
    except (ValueError, TypeError):
        pass
    return False


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
    parser = argparse.ArgumentParser(description="GAIA + RLM Agent")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--file-only", action="store_true")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--reflect", action="store_true",
                        help="Enable self-reflection verification loop")
    parser.add_argument("--output-dir", type=str, default="results/gaia_rlm")

    args = parser.parse_args()

    print("Loading GAIA dataset...")
    ds = load_dataset(GAIA_REPO, "2023_all", split="validation")
    tasks = list(ds)

    if args.level:
        tasks = [t for t in tasks if t["Level"] == args.level]
    if args.file_only:
        tasks = [t for t in tasks if t.get("file_name")]
    if not args.all_tasks:
        def needs_web(t):
            meta = t.get("Annotator Metadata", {}) or {}
            tools = meta.get("Tools", "").lower()
            return "web" in tools or "browser" in tools or "search" in tools
        tasks = [t for t in tasks if not needs_web(t)]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"  Running: {len(tasks)} tasks")

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

    results = []
    correct = 0
    total = 0

    print(f"\n{'='*60}")
    print(f"RLM Agent — {len(tasks)} tasks")
    print(f"{'='*60}")

    for i, task in enumerate(tasks):
        question = task["Question"]
        gold = task["Final answer"]
        level = task["Level"]
        has_file = bool(task.get("file_name"))

        print(f"\n[{i+1}/{len(tasks)}] L{level} {'(file)' if has_file else ''}")
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

        t0 = time.time()
        try:
            if args.reflect:
                predicted, trajectories = agent.solve_with_reflection(
                    question, file_content=file_content,
                    file_path=file_path_str,
                )
            else:
                predicted = agent.ask(
                    question, file_content=file_content,
                    file_path=file_path_str,
                )
        except Exception as e:
            predicted = f"ERROR: {e}"
        elapsed = time.time() - t0

        is_correct = check_answer(predicted, gold)
        if is_correct:
            correct += 1
        total += 1

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  Answer: {predicted[:150]}")
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

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{total} ({100*correct/max(total,1):.1f}%)")
    print(f"{'='*60}")

    for lvl in [1, 2, 3]:
        lvl_r = [r for r in results if r["level"] == lvl]
        if lvl_r:
            lc = sum(1 for r in lvl_r if r["correct"])
            print(f"  Level {lvl}: {lc}/{len(lvl_r)} ({100*lc/len(lvl_r):.1f}%)")

    avg_time = sum(r["time"] for r in results) / max(len(results), 1)
    print(f"  Avg time: {avg_time:.1f}s")

    with open(output_dir / "results.json", "w") as f:
        json.dump({"summary": {"correct": correct, "total": total},
                    "tasks": results}, f, indent=2)
    print(f"Saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
