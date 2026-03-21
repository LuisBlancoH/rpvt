"""GAIA evaluation with Claude agent + failure learning.

Runs GAIA in two passes:
  Pass 1: Blind — no lessons, records failures
  Pass 2: Learning — uses lessons from Pass 1

Usage:
    python -m rpvt.agent.eval_gaia_claude
    python -m rpvt.agent.eval_gaia_claude --pass 1
    python -m rpvt.agent.eval_gaia_claude --pass 2
    python -m rpvt.agent.eval_gaia_claude --max-tasks 20
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from rpvt.agent.claude_agent import ClaudeAgent
from rpvt.agent.file_readers import read_file


GAIA_REPO = "gaia-benchmark/GAIA"


def normalize_answer(answer):
    if answer is None:
        return ""
    s = str(answer).strip().lower()
    s = s.rstrip(".")
    # Remove common prefixes
    for prefix in ["final answer:", "answer:", "the answer is"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Remove markdown bold
    s = s.replace("**", "")
    # Normalize whitespace
    s = " ".join(s.split())
    # Normalize comma spacing
    s = s.replace(" ,", ",").replace(",  ", ", ").replace(",", ", ")
    # Re-normalize whitespace
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


def run_pass(agent, tasks, cache_dir, use_lessons=False, learn_from_failures=False):
    """Run one pass of evaluation."""
    results = []
    correct = 0
    total = 0

    for i, task in enumerate(tasks):
        task_id = task["task_id"][:8]
        question = task["Question"]
        gold = task["Final answer"]
        level = task["Level"]
        has_file = bool(task.get("file_name"))

        print(f"\n[{i+1}/{len(tasks)}] L{level} {'(file)' if has_file else ''}")
        print(f"  Q: {question[:120]}...")
        print(f"  Gold: {gold}")

        # Handle attached file
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
                elif file_content:
                    print(f"  File: {len(file_content)} chars")

        # Check for relevant lessons
        if use_lessons:
            lessons = agent.memory.retrieve(question)
            if lessons:
                print(f"  Applying {len(lessons)} lesson(s) from past failures")

        # Ask
        t0 = time.time()
        try:
            predicted = agent.ask(
                question, file_content=file_content,
                file_path=file_path_str,
                use_lessons=use_lessons,
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

        # Learn from failure
        if not is_correct and learn_from_failures:
            lesson = agent.learn_from_failure(
                question, predicted, gold, file_content
            )
            print(f"  Lesson: {lesson[:120]}")

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


def print_summary(results, label):
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"{label}: {correct}/{total} ({100*correct/max(total,1):.1f}%)")
    print(f"{'='*60}")

    for lvl in [1, 2, 3]:
        lvl_r = [r for r in results if r["level"] == lvl]
        if lvl_r:
            lc = sum(1 for r in lvl_r if r["correct"])
            print(f"  Level {lvl}: {lc}/{len(lvl_r)} ({100*lc/len(lvl_r):.1f}%)")

    file_r = [r for r in results if r["has_file"]]
    nofile_r = [r for r in results if not r["has_file"]]
    if file_r:
        fc = sum(1 for r in file_r if r["correct"])
        print(f"  With file: {fc}/{len(file_r)} ({100*fc/len(file_r):.1f}%)")
    if nofile_r:
        nc = sum(1 for r in nofile_r if r["correct"])
        print(f"  No file: {nc}/{len(nofile_r)} ({100*nc/len(nofile_r):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="GAIA + Claude + Failure Learning")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--pass", type=int, dest="run_pass", default=0,
                        help="1=blind only, 2=with lessons only, 0=both")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--file-only", action="store_true")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Include web-dependent tasks")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/gaia_claude")
    parser.add_argument("--memory-path", type=str,
                        default="~/.rpvt/gaia_failure_memory.json")

    args = parser.parse_args()

    # Load dataset
    print("Loading GAIA dataset...")
    ds = load_dataset(GAIA_REPO, "2023_all", split="validation")
    tasks = list(ds)

    if args.level:
        tasks = [t for t in tasks if t["Level"] == args.level]
        print(f"  Level {args.level}: {len(tasks)} tasks")

    if args.file_only:
        tasks = [t for t in tasks if t.get("file_name")]
        print(f"  File-only: {len(tasks)} tasks")

    if not args.all_tasks:
        def needs_web(t):
            meta = t.get("Annotator Metadata", {}) or {}
            tools = meta.get("Tools", "").lower()
            return "web" in tools or "browser" in tools or "search" in tools
        before = len(tasks)
        tasks = [t for t in tasks if not needs_web(t)]
        print(f"  No-web: {len(tasks)} tasks (removed {before - len(tasks)})")

    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    print(f"  Running: {len(tasks)} tasks")

    cache_dir = Path(args.output_dir) / "files"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize agent
    agent = ClaudeAgent(model=args.model, failure_memory_path=args.memory_path)

    # PASS 1: Blind (no lessons, learn from failures)
    if args.run_pass in (0, 1):
        agent.clear_memory()  # Start fresh
        print(f"\n{'='*60}")
        print("PASS 1: BLIND (no lessons)")
        print(f"{'='*60}")

        results1, correct1, total1 = run_pass(
            agent, tasks, cache_dir,
            use_lessons=False,
            learn_from_failures=True,
        )
        print_summary(results1, "PASS 1 (blind)")
        print(f"\nLessons recorded: {len(agent.memory)}")

        # Show lessons learned
        if agent.memory.lessons:
            print("\n--- Lessons learned ---")
            for lesson in agent.memory.lessons:
                print(f"  [{lesson['task_type']}] {lesson['lesson'][:100]}")

        with open(output_dir / "pass1_results.json", "w") as f:
            json.dump({"results": results1, "correct": correct1, "total": total1}, f, indent=2)

    # PASS 2: With lessons
    if args.run_pass in (0, 2):
        print(f"\n{'='*60}")
        print(f"PASS 2: WITH LESSONS ({len(agent.memory)} available)")
        print(f"{'='*60}")

        results2, correct2, total2 = run_pass(
            agent, tasks, cache_dir,
            use_lessons=True,
            learn_from_failures=False,
        )
        print_summary(results2, "PASS 2 (with lessons)")

        with open(output_dir / "pass2_results.json", "w") as f:
            json.dump({"results": results2, "correct": correct2, "total": total2}, f, indent=2)

    # Comparison
    if args.run_pass == 0:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"  Pass 1 (blind):        {correct1}/{total1} ({100*correct1/max(total1,1):.1f}%)")
        print(f"  Pass 2 (with lessons): {correct2}/{total2} ({100*correct2/max(total2,1):.1f}%)")
        delta = correct2 - correct1
        print(f"  Delta: {'+' if delta >= 0 else ''}{delta} tasks")

        # Which specific tasks flipped?
        if args.run_pass == 0:
            for r1, r2 in zip(results1, results2):
                if r1["correct"] != r2["correct"]:
                    direction = "FIXED" if r2["correct"] else "BROKE"
                    print(f"  [{direction}] {r1['question'][:80]}...")

        with open(output_dir / "comparison.json", "w") as f:
            json.dump({
                "pass1": {"correct": correct1, "total": total1},
                "pass2": {"correct": correct2, "total": total2},
                "delta": delta,
            }, f, indent=2)


if __name__ == "__main__":
    main()
