"""Experiment v3.3: Streaming document QA.

Process a long document chunk-by-chunk, accumulating memory, then answer
questions about any part of the document from memory alone.

This is the bridge from synthetic recall experiments to real agent memory:
- Documents are real Wikipedia articles (not synthetic templates)
- Memory accumulates continuously (not reset per document)
- Questions can be about any chunk, not just the most recent
- The model never re-reads the document — only memory is available at QA time

Uses the trained checkpoint from v3.2 (natural or synthetic) without retraining.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from rpvt.experiments.exp_v3_1_pretrained_recall import (
    build_model,
    reset_memories,
)
from rpvt.experiments.exp_v3_2_nlp_recall import _get_memory_module


def load_wikipedia_articles(n_articles=50, min_length=500, seed=42):
    """Load Wikipedia articles from WikiText-103.

    Returns list of (title, text) tuples with articles long enough to span
    multiple chunks.
    """
    print("Loading WikiText-103...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # WikiText-103 has articles separated by "= Title =" headers
    articles = []
    current_title = None
    current_text = []

    for line in wiki["text"]:
        line = line.strip()
        if line.startswith("= ") and line.endswith(" =") and not line.startswith("= ="):
            # New article
            if current_title and len(" ".join(current_text)) >= min_length:
                articles.append((current_title, " ".join(current_text)))
            current_title = line.strip("= ").strip()
            current_text = []
        elif line and current_title:
            # Skip section headers
            if not (line.startswith("= =") or line.startswith("= = =")):
                current_text.append(line)

    # Last article
    if current_title and len(" ".join(current_text)) >= min_length:
        articles.append((current_title, " ".join(current_text)))

    rng = random.Random(seed)
    rng.shuffle(articles)

    print(f"  Found {len(articles)} articles with >= {min_length} chars")
    return articles[:n_articles]


def chunk_text(text, tokenizer, chunk_size=128):
    """Split text into fixed-size token chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        ct = tokens[i:i + chunk_size]
        if len(ct) < chunk_size:
            pad_id = tokenizer.eos_token_id or 0
            ct = ct + [pad_id] * (chunk_size - len(ct))
        chunks.append(torch.tensor(ct, dtype=torch.long))
    return chunks


def extract_qa_from_article(text, tokenizer, rng, n_questions=5, chunk_size=128):
    """Extract factual QA pairs from an article using simple heuristics.

    Looks for sentences containing numbers, dates, names, and specific facts.
    Creates questions by masking the fact and asking about it.
    """
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text)
    qa_pairs = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 200:
            continue

        # Type 1: Numbers (years, quantities)
        numbers = re.findall(r'\b(\d{3,})\b', sent)
        if numbers:
            for num in numbers[:1]:
                idx = sent.index(num)
                context = sent[:idx].strip().rstrip(',').strip()
                if len(context) > 10:
                    qa_pairs.append({
                        "question": f"According to the article, what number is associated with: {context}?",
                        "answer": num,
                        "source_sentence": sent,
                        "type": "number",
                    })

        # Type 2: Named entities — "X was/is a/the Y" patterns
        # e.g. "Tatwine was an Anglo-Saxon archbishop"
        m = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was|is|are|were)\s+(?:a|an|the)\s+(.+?)(?:\.|,|;)', sent)
        if m:
            name = m.group(1)
            description = m.group(2).strip()
            if len(description) > 3 and len(description) < 60 and len(name.split()) <= 4:
                qa_pairs.append({
                    "question": f"According to the article, what was {name}?",
                    "answer": description,
                    "source_sentence": sent,
                    "type": "entity_description",
                })

        # Type 3: Location — "in/at/from X" where X is capitalized
        loc_match = re.findall(r'(?:in|at|from|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', sent)
        if loc_match:
            for loc in loc_match[:1]:
                if len(loc) > 2 and len(loc) < 30 and loc not in ("The", "This", "That", "These", "In", "At"):
                    # Use the subject of the sentence as context
                    subject = sent.split(" was ")[0] if " was " in sent else sent.split(",")[0]
                    if len(subject) > 5 and len(subject) < 80:
                        qa_pairs.append({
                            "question": f"According to the article, what location is associated with: {subject}?",
                            "answer": loc,
                            "source_sentence": sent,
                            "type": "location",
                        })

        # Type 4: "X of Y" or "X by Y" — relationships
        rel_match = re.findall(r'(?:the\s+)?([a-z]+)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', sent)
        if rel_match:
            for rel, entity in rel_match[:1]:
                if rel in ("king", "queen", "duke", "battle", "city", "university",
                           "president", "founder", "capital", "author", "bishop",
                           "son", "daughter", "wife", "husband", "leader", "head"):
                    qa_pairs.append({
                        "question": f"According to the article, who or what is the {rel} of {entity}?",
                        "answer": sent.split(f"{rel} of {entity}")[0].strip().split(",")[-1].strip() if f"{rel} of {entity}" in sent else entity,
                        "source_sentence": sent,
                        "type": "relationship",
                    })

    # Deduplicate by answer
    seen_answers = set()
    unique_qa = []
    for qa in qa_pairs:
        if qa["answer"] not in seen_answers:
            seen_answers.add(qa["answer"])
            unique_qa.append(qa)

    rng.shuffle(unique_qa)
    return unique_qa[:n_questions]


def streaming_eval(
    model, tokenizer, articles, device,
    chunk_size=128, n_questions_per_article=5,
    max_chunks_per_article=20, seed=42,
):
    """Evaluate streaming document QA.

    For each article:
    1. Process all chunks through the model, accumulating memory
    2. For each QA pair, construct a QA chunk and evaluate
    3. Measure accuracy on answer tokens

    The key difference from v3.2: memory is NOT reset between chunks of the
    same article. It accumulates continuously.
    """
    rng = random.Random(seed)
    model.eval()

    total_correct = 0
    total_tokens = 0
    total_articles = 0
    total_questions = 0
    correct_questions = 0

    results_per_article = []

    with torch.no_grad():
        for art_idx, (title, text) in enumerate(articles):
            # Chunk the article
            chunks = chunk_text(text, tokenizer, chunk_size)
            if len(chunks) > max_chunks_per_article:
                chunks = chunks[:max_chunks_per_article]

            # Extract QA pairs
            qa_pairs = extract_qa_from_article(text, tokenizer, rng,
                                                n_questions=n_questions_per_article,
                                                chunk_size=chunk_size)
            if not qa_pairs:
                continue

            # Reset memory for this article
            reset_memories(model)

            # Process all content chunks (accumulate memory)
            for chunk in chunks:
                chunk_ids = chunk.unsqueeze(0).to(device)
                model(chunk_ids)

            # Now evaluate each QA pair
            article_correct = 0
            article_total = 0
            article_q_correct = 0

            for qa in qa_pairs:
                qa_text = f"Q: {qa['question']} A: {qa['answer']}"
                qa_tokens = tokenizer.encode(qa_text, add_special_tokens=False)

                # Build answer mask
                prefix = f"Q: {qa['question']} A: "
                prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                answer_start = len(prefix_tokens)

                if len(qa_tokens) >= chunk_size:
                    qa_tokens = qa_tokens[:chunk_size]
                else:
                    pad_id = tokenizer.eos_token_id or 0
                    qa_tokens = qa_tokens + [pad_id] * (chunk_size - len(qa_tokens))

                qa_chunk = torch.tensor(qa_tokens, dtype=torch.long).unsqueeze(0).to(device)
                output = model(qa_chunk)

                # Check answer tokens
                logits = output.logits[0]
                predictions = logits[:-1].argmax(dim=-1)
                targets = qa_chunk[0, 1:]

                q_correct = 0
                q_total = 0
                for pos in range(max(0, answer_start - 1), min(len(qa_tokens) - 1, chunk_size - 1)):
                    # Only count actual answer tokens (not padding)
                    if qa_chunk[0, pos + 1].item() == (tokenizer.eos_token_id or 0):
                        break
                    q_total += 1
                    if predictions[pos].item() == targets[pos].item():
                        q_correct += 1

                article_correct += q_correct
                article_total += q_total
                total_questions += 1
                if q_total > 0 and q_correct == q_total:
                    article_q_correct += 1
                    correct_questions += 1

            total_correct += article_correct
            total_tokens += article_total
            total_articles += 1

            acc = article_correct / max(article_total, 1)
            results_per_article.append({
                "title": title,
                "n_chunks": len(chunks),
                "n_questions": len(qa_pairs),
                "token_accuracy": acc,
                "correct_tokens": article_correct,
                "total_tokens": article_total,
                "exact_match_questions": article_q_correct,
            })

            if art_idx < 10:
                print(f"\n  [{art_idx}] \"{title}\" ({len(chunks)} chunks, {len(qa_pairs)} QA)")
                print(f"    Token accuracy: {article_correct}/{article_total} = {acc:.1%}")
                for qa in qa_pairs[:2]:
                    print(f"    Q: {qa['question'][:80]}...")
                    print(f"    A: {qa['answer']}")

    overall_acc = total_correct / max(total_tokens, 1)
    overall_em = correct_questions / max(total_questions, 1)

    print(f"\n=== Streaming QA Results ===")
    print(f"  Articles: {total_articles}")
    print(f"  Questions: {total_questions}")
    print(f"  Token accuracy: {total_correct}/{total_tokens} = {overall_acc:.1%}")
    print(f"  Exact match questions: {correct_questions}/{total_questions} = {overall_em:.1%}")

    return {
        "token_accuracy": overall_acc,
        "exact_match": overall_em,
        "total_articles": total_articles,
        "total_questions": total_questions,
        "total_tokens": total_tokens,
        "correct_tokens": total_correct,
        "per_article": results_per_article,
    }


def main():
    parser = argparse.ArgumentParser(description="v3.3: Streaming document QA")

    # Model
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--memory-mode", type=str, default="cross_attn")
    parser.add_argument("--n-slots", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", type=str, default="q_proj,v_proj")
    parser.add_argument("--n-extract", type=int, default=1)

    # Checkpoint
    parser.add_argument("--load-checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (from v3.2)")

    # Eval
    parser.add_argument("--n-articles", type=int, default=50)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_3_streaming")

    # Comparison
    parser.add_argument("--no-memory", action="store_true",
                        help="Run without memory for baseline comparison")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    model, tokenizer = build_model(
        model_name=args.model_name,
        device=args.device,
        memory_layer=-1,
        memory_size=256,
        n_slots=args.n_slots,
        decay=0.999,
        gate_bias=-2.0,
        lora_rank=args.lora_rank,
        lora_targets=args.lora_targets,
        no_memory=args.no_memory,
        no_lora=False,
        init_qk_shared=False,
        n_extract=args.n_extract,
        memory_mode=args.memory_mode,
    )

    # Load checkpoint
    if not args.no_memory:
        print(f"\nLoading checkpoint from {args.load_checkpoint}...")
        checkpoint = torch.load(args.load_checkpoint, map_location=args.device,
                                weights_only=True)
        model_state = model.state_dict()
        loaded = 0
        for name, param in checkpoint.items():
            if name in model_state:
                model_state[name].copy_(param)
                loaded += 1
        print(f"  Loaded {loaded} parameter tensors")

    # Load articles
    articles = load_wikipedia_articles(
        n_articles=args.n_articles,
        min_length=500,
        seed=args.seed,
    )

    # Run streaming eval
    results = streaming_eval(
        model, tokenizer, articles, args.device,
        chunk_size=args.chunk_size,
        n_questions_per_article=args.n_questions,
        max_chunks_per_article=args.max_chunks,
        seed=args.seed,
    )

    results["config"] = vars(args)

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
