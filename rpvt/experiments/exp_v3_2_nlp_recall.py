"""Experiment v3.2: Natural language recall with pretrained model + Hopfield memory.

Replaces synthetic token recall with SQuAD-based extractive QA. The model sees
a passage in early chunks and must answer questions about it in a later chunk,
with per-chunk processing so memory is the only cross-chunk information channel.

Key improvements over v3.1 (synthetic):
  - Loss computed ONLY on answer tokens → no loss dilution
  - Multiple QA pairs per passage → dense memory-dependent signal
  - Natural language → plays to pretrained model's strengths
  - Directly scales to harder tasks

Architecture: Frozen Qwen2.5-3B + LoRA + HopfieldMemory at layer 18/36.
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

# Use writable HF cache if default is read-only
if not os.access(os.environ.get("HF_HOME", "/workspace/.hf_home"), os.W_OK):
    os.environ["HF_HOME"] = "/tmp/hf_home"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from rpvt.model.hopfield_memory import HopfieldMemory

# Reuse model infrastructure from v3.1
from rpvt.experiments.exp_v3_1_pretrained_recall import (
    MemoryWrapper,
    reset_memories,
    set_persistent_grad,
    detach_memory_state,
    build_model,
    get_memory_params,
    get_lora_params,
)


def _random_name(rng):
    """Generate a unique random name (not in any training corpus)."""
    syllables = ["bal", "cor", "del", "fen", "gor", "hal", "ith", "jan",
                 "kel", "lor", "mav", "nex", "ost", "pyr", "quel", "ren",
                 "sal", "tor", "ush", "vel", "wen", "xar", "yel", "zin",
                 "bri", "cra", "dru", "fal", "gri", "hov", "isk", "jol"]
    n = rng.randint(2, 3)
    first = "".join(rng.sample(syllables, n)).capitalize()
    n = rng.randint(2, 3)
    last = "".join(rng.sample(syllables, n)).capitalize()
    return f"{first} {last}"


def _random_word(rng, min_len=5, max_len=10):
    """Generate a random but pronounceable word."""
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    length = rng.randint(min_len, max_len)
    word = ""
    for i in range(length):
        if i % 2 == 0:
            word += rng.choice(consonants)
        else:
            word += rng.choice(vowels)
    return word


def _generate_synthetic_facts(rng, n_docs, max_qa_pairs):
    """Generate synthetic passages with truly unique facts per document.

    Every entity name, city, organization, etc. is randomly generated
    to prevent LoRA from memorizing the answer distribution.
    The ONLY way to predict the answer is to have read the passage.
    """
    docs = []
    for _ in range(n_docs):
        name = _random_name(rng)
        first = name.split()[0]
        city = _random_word(rng, 5, 9).capitalize()
        org = f"the {_random_word(rng, 5, 8).capitalize()} {rng.choice(['Institute', 'Foundation', 'Society', 'Bureau', 'Council'])}"
        field = f"{_random_word(rng, 4, 7)} {rng.choice(['theory', 'dynamics', 'analysis', 'systems', 'engineering'])}"
        # Use truly random numbers
        code_a = rng.randint(100, 999)
        code_b = rng.randint(100, 999)
        year = rng.randint(1800, 2020)

        passage = (
            f"{name} was a researcher at {org} in {city}. "
            f"Their work on {field} produced result code {code_a}. "
            f"The project started in {year} and generated output code {code_b}. "
            f"{first} specialized in {field} at the {city} campus."
        )

        all_qas = [
            {"question": f"Where did {name} work?", "answer": city},
            {"question": f"What organization was {name} part of?", "answer": org},
            {"question": f"What was {name}'s result code?", "answer": str(code_a)},
            {"question": f"When did {name}'s project start?", "answer": str(year)},
            {"question": f"What was {name}'s output code?", "answer": str(code_b)},
            {"question": f"What field did {name} work in?", "answer": field},
        ]
        selected_qas = rng.sample(all_qas, min(max_qa_pairs, len(all_qas)))
        docs.append((passage, selected_qas))

    return docs


def _generate_single_person_facts(rng):
    """Generate a single person's passage and QA pairs."""
    name = _random_name(rng)
    first = name.split()[0]
    city = _random_word(rng, 5, 9).capitalize()
    org = f"the {_random_word(rng, 5, 8).capitalize()} {rng.choice(['Institute', 'Foundation', 'Society', 'Bureau', 'Council'])}"
    field = f"{_random_word(rng, 4, 7)} {rng.choice(['theory', 'dynamics', 'analysis', 'systems', 'engineering'])}"
    code_a = rng.randint(100, 999)
    code_b = rng.randint(100, 999)
    year = rng.randint(1800, 2020)

    passage = (
        f"{name} was a researcher at {org} in {city}. "
        f"Their work on {field} produced result code {code_a}. "
        f"The project started in {year} and generated output code {code_b}. "
        f"{first} specialized in {field} at the {city} campus."
    )

    all_qas = [
        {"question": f"Where did {name} work?", "answer": city},
        {"question": f"What organization was {name} part of?", "answer": org},
        {"question": f"What was {name}'s result code?", "answer": str(code_a)},
        {"question": f"When did {name}'s project start?", "answer": str(year)},
        {"question": f"What was {name}'s output code?", "answer": str(code_b)},
        {"question": f"What field did {name} work in?", "answer": field},
    ]
    return passage, all_qas


def _generate_confusable_passage_facts(rng, n_docs, max_qa_pairs_per_passage=3):
    """Generate documents with TWO passages that SHARE some facts.

    Tests whether memory binds facts to entities (not just keyword matching).
    Two people may share a city, organization, or field — but have different
    codes and years. The model must use the person's name to retrieve the
    right facts, not just match on shared attributes.

    Returns list of tuples: (passage_a, passage_b, combined_qas)
    """
    docs = []
    for _ in range(n_docs):
        # Generate two people
        name_a = _random_name(rng)
        name_b = _random_name(rng)
        first_a = name_a.split()[0]
        first_b = name_b.split()[0]

        # Shared attributes (pick 1-2 to share)
        shared_city = _random_word(rng, 5, 9).capitalize()
        shared_org = f"the {_random_word(rng, 5, 8).capitalize()} {rng.choice(['Institute', 'Foundation', 'Society', 'Bureau', 'Council'])}"
        shared_field = f"{_random_word(rng, 4, 7)} {rng.choice(['theory', 'dynamics', 'analysis', 'systems', 'engineering'])}"

        n_shared = rng.randint(1, 2)  # share 1-2 attributes
        share_what = rng.sample(["city", "org", "field"], n_shared)

        city_a = shared_city if "city" in share_what else _random_word(rng, 5, 9).capitalize()
        city_b = shared_city if "city" in share_what else _random_word(rng, 5, 9).capitalize()
        org_a = shared_org if "org" in share_what else f"the {_random_word(rng, 5, 8).capitalize()} {rng.choice(['Institute', 'Foundation', 'Society', 'Bureau', 'Council'])}"
        org_b = shared_org if "org" in share_what else f"the {_random_word(rng, 5, 8).capitalize()} {rng.choice(['Institute', 'Foundation', 'Society', 'Bureau', 'Council'])}"
        field_a = shared_field if "field" in share_what else f"{_random_word(rng, 4, 7)} {rng.choice(['theory', 'dynamics', 'analysis', 'systems', 'engineering'])}"
        field_b = shared_field if "field" in share_what else f"{_random_word(rng, 4, 7)} {rng.choice(['theory', 'dynamics', 'analysis', 'systems', 'engineering'])}"

        # Unique numeric facts (always different)
        code_a1, code_a2 = rng.randint(100, 999), rng.randint(100, 999)
        code_b1, code_b2 = rng.randint(100, 999), rng.randint(100, 999)
        year_a = rng.randint(1800, 2020)
        year_b = rng.randint(1800, 2020)

        passage_a = (
            f"{name_a} was a researcher at {org_a} in {city_a}. "
            f"Their work on {field_a} produced result code {code_a1}. "
            f"The project started in {year_a} and generated output code {code_a2}. "
            f"{first_a} specialized in {field_a} at the {city_a} campus."
        )
        passage_b = (
            f"{name_b} was a researcher at {org_b} in {city_b}. "
            f"Their work on {field_b} produced result code {code_b1}. "
            f"The project started in {year_b} and generated output code {code_b2}. "
            f"{first_b} specialized in {field_b} at the {city_b} campus."
        )

        qas_a = [
            {"question": f"Where did {name_a} work?", "answer": city_a},
            {"question": f"What organization was {name_a} part of?", "answer": org_a},
            {"question": f"What was {name_a}'s result code?", "answer": str(code_a1)},
            {"question": f"When did {name_a}'s project start?", "answer": str(year_a)},
            {"question": f"What was {name_a}'s output code?", "answer": str(code_a2)},
            {"question": f"What field did {name_a} work in?", "answer": field_a},
        ]
        qas_b = [
            {"question": f"Where did {name_b} work?", "answer": city_b},
            {"question": f"What organization was {name_b} part of?", "answer": org_b},
            {"question": f"What was {name_b}'s result code?", "answer": str(code_b1)},
            {"question": f"When did {name_b}'s project start?", "answer": str(year_b)},
            {"question": f"What was {name_b}'s output code?", "answer": str(code_b2)},
            {"question": f"What field did {name_b} work in?", "answer": field_b},
        ]

        selected_a = rng.sample(qas_a, min(max_qa_pairs_per_passage, len(qas_a)))
        selected_b = rng.sample(qas_b, min(max_qa_pairs_per_passage, len(qas_b)))

        combined_qas = []
        for qa_a, qa_b in zip(selected_a, selected_b):
            combined_qas.append(qa_a)
            combined_qas.append(qa_b)
        combined_qas.extend(selected_a[len(selected_b):])
        combined_qas.extend(selected_b[len(selected_a):])

        docs.append((passage_a, passage_b, combined_qas))
    return docs


def _generate_n_passage_facts(rng, n_docs, n_passages=5, max_qa_pairs_per_passage=2):
    """Generate documents with N passages about different people.

    Tests memory capacity and attention selectivity — the model must pick
    the right slot out of N passage slots when answering questions.

    Returns list of tuples: (list_of_passages, combined_qas)
    """
    docs = []
    for _ in range(n_docs):
        passages = []
        all_selected_qas = []

        for _ in range(n_passages):
            passage, qas = _generate_single_person_facts(rng)
            passages.append(passage)
            selected = rng.sample(qas, min(max_qa_pairs_per_passage, len(qas)))
            all_selected_qas.extend(selected)

        # Shuffle QA pairs so questions about different people are interleaved
        rng.shuffle(all_selected_qas)

        docs.append((passages, all_selected_qas))
    return docs


def _generate_multi_passage_facts(rng, n_docs, max_qa_pairs_per_passage=3):
    """Generate documents with TWO passages about different people.

    Each document has two passages (A and B), each about a different person.
    QA pairs are interleaved from both passages, testing whether memory can
    keep facts from different passages separate.

    Returns list of tuples: (passage_a, passage_b, combined_qas)
    """
    docs = []
    for _ in range(n_docs):
        passage_a, qas_a = _generate_single_person_facts(rng)
        passage_b, qas_b = _generate_single_person_facts(rng)

        # Select QA pairs from each passage
        selected_a = rng.sample(qas_a, min(max_qa_pairs_per_passage, len(qas_a)))
        selected_b = rng.sample(qas_b, min(max_qa_pairs_per_passage, len(qas_b)))

        # Interleave QA pairs from both passages
        combined_qas = []
        for qa_a, qa_b in zip(selected_a, selected_b):
            combined_qas.append(qa_a)
            combined_qas.append(qa_b)
        # Add any remaining if lengths differ
        combined_qas.extend(selected_a[len(selected_b):])
        combined_qas.extend(selected_b[len(selected_a):])

        docs.append((passage_a, passage_b, combined_qas))
    return docs


class SQuADRecallDataset(Dataset):
    """Recall task with per-chunk processing.

    Supports three data sources:
    - "squad": SQuAD v2 passages (model may know answers from pretraining)
    - "synthetic": Generated passages with novel facts (answers require memory)
    - "synthetic_multi": Two passages per doc, QA about both (tests interference)

    Document structure (single passage):
      - 1+ passage chunks (context, tokenized)
      - gap_min..gap_max filler chunks (WikiText paragraphs)
      - 1 QA chunk: "Q: <question> A: <answer> Q: ... A: ..."

    Document structure (multi-passage):
      - 1+ passage A chunks
      - gap_min..gap_max filler chunks
      - 1+ passage B chunks
      - gap_min..gap_max filler chunks
      - 1 QA chunk with interleaved questions about both passages

    Loss is computed ONLY on answer tokens (after each "A:").
    The passage is only accessible via memory — per-chunk processing prevents
    cross-chunk attention. Every answer token requires memory.
    """

    def __init__(
        self,
        tokenizer,
        split="train",
        n_docs=500,
        chunk_size=128,
        gap_range=(2, 6),
        max_qa_pairs=3,
        seed=42,
        data_source="synthetic",
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.max_qa_pairs = max_qa_pairs
        self.n_pairs = max_qa_pairs  # for gate analysis compat
        self.data_source = data_source

        rng = random.Random(seed)

        if data_source == "squad":
            from datasets import load_dataset
            print(f"  Loading SQuAD v2 ({split})...")
            squad = load_dataset("rajpurkar/squad_v2", split=split)

            context_to_qas = defaultdict(list)
            for ex in squad:
                if ex["answers"]["text"]:
                    context_to_qas[ex["context"]].append({
                        "question": ex["question"],
                        "answer": ex["answers"]["text"][0],
                    })

            contexts = []
            for ctx, qas in context_to_qas.items():
                if len(qas) >= max_qa_pairs:
                    ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
                    if len(ctx_tokens) <= chunk_size * 4:
                        contexts.append((ctx, qas, ctx_tokens))

            print(f"  Found {len(contexts)} passages with >= {max_qa_pairs} QA pairs")
            rng.shuffle(contexts)
            passage_qas = [(ctx, qas) for ctx, qas, _ in contexts[:n_docs]]

        elif data_source == "synthetic":
            print(f"  Generating {n_docs} synthetic passages...")
            generated = _generate_synthetic_facts(rng, n_docs, max_qa_pairs)
            passage_qas = generated

        elif data_source == "synthetic_multi":
            print(f"  Generating {n_docs} multi-passage documents...")
            multi_docs = _generate_multi_passage_facts(
                rng, n_docs, max_qa_pairs_per_passage=max_qa_pairs
            )
            # Will be handled separately below
            passage_qas = None

        elif data_source == "synthetic_confusable":
            print(f"  Generating {n_docs} confusable multi-passage documents...")
            multi_docs = _generate_confusable_passage_facts(
                rng, n_docs, max_qa_pairs_per_passage=max_qa_pairs
            )
            passage_qas = None

        elif data_source.startswith("synthetic_n_"):
            # e.g. "synthetic_n_5" for 5 passages
            n_passages = int(data_source.split("_")[-1])
            # Fewer QA per person to keep QA chunk manageable
            qa_per_person = min(max_qa_pairs, 2)
            print(f"  Generating {n_docs} documents with {n_passages} passages ({qa_per_person} QA each)...")
            n_passage_docs = _generate_n_passage_facts(
                rng, n_docs, n_passages=n_passages,
                max_qa_pairs_per_passage=qa_per_person,
            )
            # Will be handled separately below
            passage_qas = None
            multi_docs = None

        else:
            raise ValueError(f"Unknown data_source: {data_source}")

        # Load filler text (WikiText)
        from datasets import load_dataset
        print(f"  Loading WikiText for filler...")
        wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        filler_texts = [t for t in wiki["text"] if len(t.strip()) > 100]
        print(f"  {len(filler_texts)} filler paragraphs available")

        self.documents = []

        if data_source in ("synthetic_multi", "synthetic_confusable"):
            # Multi-passage: [PassageA] [filler] [PassageB] [filler] [QA about both]
            for doc_idx in range(len(multi_docs)):
                passage_a, passage_b, combined_qas = multi_docs[doc_idx]
                gap_a = rng.randint(gap_range[0], gap_range[1])
                gap_b = rng.randint(gap_range[0], gap_range[1])

                def _make_chunks(text):
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    chunks = []
                    for i in range(0, len(tokens), chunk_size):
                        ct = tokens[i:i + chunk_size]
                        if len(ct) < chunk_size:
                            pad_id = tokenizer.eos_token_id or 0
                            ct = ct + [pad_id] * (chunk_size - len(ct))
                        chunks.append(torch.tensor(ct, dtype=torch.long))
                    return chunks

                def _make_filler(n):
                    chunks = []
                    for _ in range(n):
                        ft = rng.choice(filler_texts)
                        ft_tok = tokenizer.encode(ft, add_special_tokens=False)
                        if len(ft_tok) >= chunk_size:
                            start = rng.randint(0, len(ft_tok) - chunk_size)
                            ct = ft_tok[start:start + chunk_size]
                        else:
                            pad_id = tokenizer.eos_token_id or 0
                            ct = ft_tok + [pad_id] * (chunk_size - len(ft_tok))
                        chunks.append(torch.tensor(ct, dtype=torch.long))
                    return chunks

                passage_a_chunks = _make_chunks(passage_a)
                filler_a_chunks = _make_filler(gap_a)
                passage_b_chunks = _make_chunks(passage_b)
                filler_b_chunks = _make_filler(gap_b)

                # Build QA chunk
                qa_text = ""
                for qa in combined_qas:
                    qa_text += f" Q: {qa['question']} A: {qa['answer']}"
                qa_text = qa_text.strip()

                qa_tokens = tokenizer.encode(qa_text, add_special_tokens=False)
                answer_mask = self._build_answer_mask(qa_text, combined_qas, tokenizer, chunk_size)

                if len(qa_tokens) >= chunk_size:
                    qa_tokens = qa_tokens[:chunk_size]
                else:
                    pad_id = tokenizer.eos_token_id or 0
                    qa_tokens = qa_tokens + [pad_id] * (chunk_size - len(qa_tokens))
                qa_chunk = torch.tensor(qa_tokens, dtype=torch.long)

                all_chunks = (passage_a_chunks + filler_a_chunks +
                              passage_b_chunks + filler_b_chunks + [qa_chunk])
                n_passage = len(passage_a_chunks) + len(passage_b_chunks)

                self.documents.append({
                    "chunks": all_chunks,
                    "answer_mask": answer_mask,
                    "qa_pairs": combined_qas,
                    "gap": gap_a + gap_b,
                    "n_passage_chunks": n_passage,
                    "context": f"{passage_a}\n---\n{passage_b}",
                })
        elif data_source.startswith("synthetic_n_"):
            # N-passage: [P1] [filler] [P2] [filler] ... [PN] [filler] [QA about all]
            for doc_idx in range(len(n_passage_docs)):
                passages, combined_qas = n_passage_docs[doc_idx]

                def _make_chunks(text):
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    chunks = []
                    for i in range(0, len(tokens), chunk_size):
                        ct = tokens[i:i + chunk_size]
                        if len(ct) < chunk_size:
                            pad_id = tokenizer.eos_token_id or 0
                            ct = ct + [pad_id] * (chunk_size - len(ct))
                        chunks.append(torch.tensor(ct, dtype=torch.long))
                    return chunks

                def _make_filler(n):
                    chunks = []
                    for _ in range(n):
                        ft = rng.choice(filler_texts)
                        ft_tok = tokenizer.encode(ft, add_special_tokens=False)
                        if len(ft_tok) >= chunk_size:
                            start = rng.randint(0, len(ft_tok) - chunk_size)
                            ct = ft_tok[start:start + chunk_size]
                        else:
                            pad_id = tokenizer.eos_token_id or 0
                            ct = ft_tok + [pad_id] * (chunk_size - len(ft_tok))
                        chunks.append(torch.tensor(ct, dtype=torch.long))
                    return chunks

                all_chunks = []
                n_passage_total = 0
                total_gap = 0
                for passage in passages:
                    p_chunks = _make_chunks(passage)
                    gap = rng.randint(gap_range[0], gap_range[1])
                    f_chunks = _make_filler(gap)
                    all_chunks.extend(p_chunks)
                    all_chunks.extend(f_chunks)
                    n_passage_total += len(p_chunks)
                    total_gap += gap

                # Build QA chunk
                qa_text = ""
                for qa in combined_qas:
                    qa_text += f" Q: {qa['question']} A: {qa['answer']}"
                qa_text = qa_text.strip()

                qa_tokens = tokenizer.encode(qa_text, add_special_tokens=False)
                answer_mask = self._build_answer_mask(qa_text, combined_qas, tokenizer, chunk_size)

                if len(qa_tokens) >= chunk_size:
                    qa_tokens = qa_tokens[:chunk_size]
                else:
                    pad_id = tokenizer.eos_token_id or 0
                    qa_tokens = qa_tokens + [pad_id] * (chunk_size - len(qa_tokens))
                qa_chunk = torch.tensor(qa_tokens, dtype=torch.long)

                all_chunks.append(qa_chunk)

                self.documents.append({
                    "chunks": all_chunks,
                    "answer_mask": answer_mask,
                    "qa_pairs": combined_qas,
                    "gap": total_gap,
                    "n_passage_chunks": n_passage_total,
                    "context": "\n---\n".join(passages),
                })
        else:
            # Single-passage: [Passage] [filler] [QA]
            for doc_idx in range(len(passage_qas)):
                ctx_text, selected_qas = passage_qas[doc_idx]
                ctx_tokens = tokenizer.encode(ctx_text, add_special_tokens=False)
                gap = rng.randint(gap_range[0], gap_range[1])

                # Select QA pairs (may already be selected for synthetic)
                if len(selected_qas) > max_qa_pairs:
                    selected_qas = rng.sample(selected_qas, max_qa_pairs)

                # Build passage chunks
                passage_chunks = []
                for i in range(0, len(ctx_tokens), chunk_size):
                    chunk_tokens = ctx_tokens[i:i + chunk_size]
                    if len(chunk_tokens) < chunk_size:
                        # Pad with eos token
                        pad_id = tokenizer.eos_token_id or 0
                        chunk_tokens = chunk_tokens + [pad_id] * (chunk_size - len(chunk_tokens))
                    passage_chunks.append(torch.tensor(chunk_tokens, dtype=torch.long))

                # Build filler chunks
                filler_chunks = []
                for _ in range(gap):
                    filler_text = rng.choice(filler_texts)
                    filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)
                    if len(filler_tokens) >= chunk_size:
                        start = rng.randint(0, len(filler_tokens) - chunk_size)
                        chunk_tokens = filler_tokens[start:start + chunk_size]
                    else:
                        pad_id = tokenizer.eos_token_id or 0
                        chunk_tokens = filler_tokens + [pad_id] * (chunk_size - len(filler_tokens))
                    filler_chunks.append(torch.tensor(chunk_tokens, dtype=torch.long))

                # Build QA chunk with answer masks
                qa_text = ""
                for qa in selected_qas:
                    qa_text += f" Q: {qa['question']} A: {qa['answer']}"
                qa_text = qa_text.strip()

                qa_tokens = tokenizer.encode(qa_text, add_special_tokens=False)

                # Find answer token positions by looking for "A:" markers
                answer_mask = self._build_answer_mask(qa_text, selected_qas, tokenizer, chunk_size)

                if len(qa_tokens) >= chunk_size:
                    qa_tokens = qa_tokens[:chunk_size]
                else:
                    pad_id = tokenizer.eos_token_id or 0
                    qa_tokens = qa_tokens + [pad_id] * (chunk_size - len(qa_tokens))

                qa_chunk = torch.tensor(qa_tokens, dtype=torch.long)

                # Combine all chunks
                all_chunks = passage_chunks + filler_chunks + [qa_chunk]

                self.documents.append({
                    "chunks": all_chunks,
                    "answer_mask": answer_mask,  # (chunk_size,) binary mask for QA chunk
                    "qa_pairs": selected_qas,
                    "gap": gap,
                    "n_passage_chunks": len(passage_chunks),
                    "context": ctx_text,
                })

        # For compat with v3.1 training loop
        self.doc_sequences = self.documents

        print(f"  NLRecall({data_source}): {len(self.documents)} docs, "
              f"gap range {gap_range}, {max_qa_pairs} QA pairs/doc, "
              f"chunk_size={chunk_size}")

    def _build_answer_mask(self, qa_text, selected_qas, tokenizer, chunk_size):
        """Build binary mask marking answer tokens in the QA chunk.

        The mask marks tokens that are part of answer text (after "A:").
        Loss is computed only on these tokens.
        """
        # Strategy: tokenize the full QA text, then find answer spans
        # by tokenizing up to each answer boundary
        full_tokens = tokenizer.encode(qa_text, add_special_tokens=False)
        mask = torch.zeros(chunk_size, dtype=torch.float32)

        # Find each answer's token positions
        search_from = 0
        for qa in selected_qas:
            # Find "A: <answer>" in the text
            answer_marker = f"A: {qa['answer']}"
            marker_pos = qa_text.find(answer_marker, search_from)
            if marker_pos == -1:
                continue

            # Tokenize up to the answer start (after "A: ")
            prefix = qa_text[:marker_pos + 3]  # include "A: "
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            answer_start = len(prefix_tokens)

            # Tokenize up to end of answer
            suffix = qa_text[:marker_pos + len(answer_marker)]
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
            answer_end = len(suffix_tokens)

            # Mark answer tokens (shifted by 1 for next-token prediction)
            # When computing loss, logits[t] predicts token[t+1],
            # so to get loss on answer tokens, we mark positions t where token[t+1] is answer
            for pos in range(max(0, answer_start - 1), min(answer_end - 1, chunk_size)):
                mask[pos] = 1.0

            search_from = marker_pos + len(answer_marker)

        return mask

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]


def _get_memory_module(model):
    """Get the HopfieldMemory or MemoryBank module from the model."""
    from rpvt.model.cross_attention_memory import MemoryBank
    for module in model.modules():
        if isinstance(module, HopfieldMemory):
            return module
        if isinstance(module, MemoryBank):
            return module
    return None


def _compute_attention_supervision_loss(mem, passage_slot_indices, device):
    """Compute attention supervision loss on Hopfield memory.

    After the QA chunk is processed, mem.last_attn_weights contains the
    attention distribution over K_mem slots. We push this attention toward
    slots that contain passage content (not filler).

    This avoids the collapse problem of cosine similarity loss — the target
    is a distribution over discrete slots, not a continuous direction.

    Args:
        mem: HopfieldMemory module with last_attn_weights set
        passage_slot_indices: list of int, which K_mem slots hold passage content
        device: torch device

    Returns:
        Scalar loss (KL divergence from target attention distribution).
    """
    if mem.last_attn_weights is None or not passage_slot_indices:
        return torch.tensor(0.0, device=device)

    # attn_weights: (batch, qa_tokens, n_slots) from QA chunk retrieval
    attn_weights = mem.last_attn_weights  # already has gradients from forward

    n_slots = attn_weights.shape[-1]

    # Build target: uniform over passage slots, zero on everything else
    target = torch.zeros(n_slots, device=device)
    for idx in passage_slot_indices:
        if idx < n_slots:
            target[idx] = 1.0

    if target.sum() < 1e-8:
        return torch.tensor(0.0, device=device)

    target = target / target.sum()  # normalize to distribution

    # Average attention across batch and qa tokens → (n_slots,)
    # Then compute KL(target || attn) to push attn toward passage slots
    mean_attn = attn_weights.mean(dim=(0, 1))  # (n_slots,)

    # Cross-entropy loss: -sum(target * log(attn + eps))
    # This is simpler and more stable than KL divergence
    loss = -(target * torch.log(mean_attn + 1e-8)).sum()

    return loss


def train_and_eval(
    model, train_dataset, eval_dataset, device,
    num_epochs=10, lr_memory=1e-3, lr_lora=2e-4,
    log_every=50, two_phase=False, output_dir="results",
    retrieval_loss_weight=0.0,
):
    """Train with answer-only loss on QA chunks."""
    memory_params = get_memory_params(model)
    lora_params = get_lora_params(model)

    if two_phase:
        for name, param in model.named_parameters():
            if name in memory_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Two-phase: Phase 1 — {n_train:,} memory params trainable")

    # Build optimizer
    param_groups = []
    seen_ids = set()

    mem_p = []
    for n, p in model.named_parameters():
        if p.requires_grad and n in memory_params and id(p) not in seen_ids:
            mem_p.append(p)
            seen_ids.add(id(p))

    lora_p = []
    for n, p in model.named_parameters():
        if p.requires_grad and n in lora_params and id(p) not in seen_ids:
            lora_p.append(p)
            seen_ids.add(id(p))

    other_p = []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen_ids:
            other_p.append(p)
            seen_ids.add(id(p))

    if mem_p:
        param_groups.append({"params": mem_p, "lr": lr_memory})
    if lora_p:
        param_groups.append({"params": lora_p, "lr": lr_lora})
    if other_p:
        param_groups.append({"params": other_p, "lr": lr_lora})

    if not param_groups:
        print("  WARNING: No trainable parameters!")
        return evaluate(model, eval_dataset, device)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = len(train_dataset.documents) * num_epochs

    def lr_schedule(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    phase_1_end = num_epochs // 2 if two_phase else 0
    phase_1_done = False

    print(f"\nTraining: {total_steps} steps, {num_epochs} epochs"
          f"{f', retrieval_weight={retrieval_loss_weight}' if retrieval_loss_weight > 0 else ''}")
    model.train()
    global_step = 0
    train_losses = []
    answer_losses = []
    ret_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        # Phase transition
        if two_phase and epoch == phase_1_end and not phase_1_done:
            print(f"\n  === PHASE 2: Unfreezing LoRA ===")
            for name, param in model.named_parameters():
                if name in memory_params or name in lora_params:
                    param.requires_grad = True

            seen_ids_p2 = set()
            mem_p = []
            for n, p in model.named_parameters():
                if p.requires_grad and n in memory_params and id(p) not in seen_ids_p2:
                    mem_p.append(p)
                    seen_ids_p2.add(id(p))
            lora_p = []
            for n, p in model.named_parameters():
                if p.requires_grad and n in lora_params and id(p) not in seen_ids_p2:
                    lora_p.append(p)
                    seen_ids_p2.add(id(p))
            param_groups = []
            if mem_p:
                param_groups.append({"params": mem_p, "lr": lr_memory * 0.1})
            if lora_p:
                param_groups.append({"params": lora_p, "lr": lr_lora})

            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
            remaining_steps = total_steps - global_step
            def lr_schedule_p2(step, _rem=remaining_steps):
                warmup = 50
                if step < warmup:
                    return step / warmup
                progress = (step - warmup) / max(_rem - warmup, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_p2)

            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  {n_train:,} params trainable (memory + LoRA)")
            phase_1_done = True

        epoch_order = list(range(len(train_dataset.documents)))
        random.shuffle(epoch_order)

        for doc_idx in epoch_order:
            doc = train_dataset.documents[doc_idx]
            chunks = doc["chunks"]
            n_chunks = len(chunks)
            n_passage = doc["n_passage_chunks"]
            answer_mask = doc["answer_mask"].to(device)

            # Reset memory and enable gradient persistence
            reset_memories(model)
            set_persistent_grad(model, True)

            # For attention supervision: track which K_mem slots hold passage content
            mem = _get_memory_module(model)
            passage_slot_indices = []  # which write_ptr slots got passage data

            # Process each chunk independently
            doc_loss = torch.tensor(0.0, device=device)
            qa_loss = torch.tensor(0.0, device=device)
            n_loss_chunks = 0

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)  # (1, chunk_size)
                is_qa_chunk = (chunk_idx == n_chunks - 1)
                is_passage = chunk_idx < n_passage

                # Track write_ptr before forward to know which slots this chunk writes to
                if retrieval_loss_weight > 0 and mem is not None and is_passage:
                    n_ext = getattr(mem, 'n_extract', 1)
                    for ei in range(n_ext):
                        slot_idx = ((mem.write_ptr + ei) % mem.n_slots).item()
                        passage_slot_indices.append(slot_idx)

                output = model(chunk_ids)

                if is_qa_chunk:
                    # Answer-only loss on QA chunk
                    logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                    targets = chunk_ids[:, 1:].reshape(-1)
                    per_token_loss = F.cross_entropy(logits, targets, reduction='none')

                    # Apply answer mask (shifted by 1 for next-token prediction)
                    mask = answer_mask[:-1]  # (chunk_size - 1,)
                    n_answer_tokens = mask.sum().clamp(min=1)
                    chunk_loss = (per_token_loss * mask).sum() / n_answer_tokens
                    qa_loss = qa_loss + chunk_loss
                    doc_loss = doc_loss + chunk_loss
                    n_loss_chunks += 1
                else:
                    # Standard LM loss on passage/filler chunks
                    logits = output.logits[:, :-1].reshape(-1, output.logits.size(-1))
                    targets = chunk_ids[:, 1:].reshape(-1)
                    chunk_loss = F.cross_entropy(logits, targets)
                    doc_loss = doc_loss + chunk_loss * 0.1  # downweight non-QA loss
                    n_loss_chunks += 1

            # Average loss
            doc_loss = doc_loss / max(n_loss_chunks, 1)

            # Add attention supervision loss
            ret_loss = torch.tensor(0.0, device=device)
            if retrieval_loss_weight > 0 and mem is not None:
                ret_loss = _compute_attention_supervision_loss(
                    mem, passage_slot_indices, device
                )
                doc_loss = doc_loss + retrieval_loss_weight * ret_loss

            optimizer.zero_grad()
            doc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            # Detach memory state
            detach_memory_state(model)
            set_persistent_grad(model, False)

            train_losses.append(doc_loss.item())
            answer_losses.append(qa_loss.item())
            ret_losses.append(ret_loss.item() if isinstance(ret_loss, torch.Tensor) else ret_loss)
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = sum(train_losses[-log_every:]) / log_every
                avg_qa = sum(answer_losses[-log_every:]) / log_every
                avg_ret = sum(ret_losses[-log_every:]) / log_every
                elapsed = time.time() - start_time
                ret_info = f", ret_loss={avg_ret:.4f}" if retrieval_loss_weight > 0 else ""
                print(f"  step {global_step}/{total_steps}, "
                      f"loss={avg_loss:.4f}, qa_loss={avg_qa:.4f}{ret_info}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, {elapsed:.0f}s")

        # Evaluate at end of each epoch
        print(f"\n  === Epoch {epoch + 1}/{num_epochs} eval ===")
        eval_results = evaluate(model, eval_dataset, device, verbose=False)
        attn_info = ""
        if eval_results.get("attn_on_passage") is not None:
            attn_info = f", attn_passage={eval_results['attn_on_passage']:.1%}"
        print(f"  Token accuracy: {eval_results['token_accuracy']:.1%}, "
              f"Exact match: {eval_results['exact_match']:.1%}{attn_info}")
        model.train()

    return evaluate(model, eval_dataset, device, verbose=True)


def evaluate(model, dataset, device, verbose=True, n_debug=10):
    """Evaluate QA recall accuracy.

    Metrics:
      - Token accuracy: fraction of answer tokens predicted correctly (top-1)
      - Exact match: fraction of QA pairs where ALL answer tokens are correct
    """
    model.eval()

    total_answer_tokens = 0
    correct_answer_tokens = 0
    total_pairs = 0
    exact_match_pairs = 0
    debug_count = 0

    # Attention analysis: how much attention lands on passage slots
    attn_on_passage_list = []

    with torch.no_grad():
        for doc_idx, doc in enumerate(dataset.documents):
            chunks = doc["chunks"]
            n_chunks = len(chunks)
            answer_mask = doc["answer_mask"]
            n_passage = doc["n_passage_chunks"]
            gap = doc["gap"]

            reset_memories(model)

            mem = _get_memory_module(model)
            passage_slot_indices = []

            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                is_passage = chunk_idx < n_passage

                # Track passage slots (n_extract slots per chunk)
                if mem is not None and is_passage:
                    n_ext = getattr(mem, 'n_extract', 1)
                    for ei in range(n_ext):
                        slot_idx = ((mem.write_ptr + ei) % mem.n_slots).item()
                        passage_slot_indices.append(slot_idx)

                output = model(chunk_ids)

                is_qa_chunk = (chunk_idx == n_chunks - 1)

                if is_qa_chunk:
                    # Measure attention on passage slots (Hopfield memory only)
                    if mem is not None and hasattr(mem, 'last_attn_weights') and mem.last_attn_weights is not None and passage_slot_indices:
                        attn = mem.last_attn_weights.mean(dim=(0, 1))  # (n_slots,)
                        passage_attn = sum(attn[s].item() for s in passage_slot_indices if s < len(attn))
                        attn_on_passage_list.append(passage_attn)
                    # Check answer tokens
                    logits = output.logits[0]  # (chunk_size, vocab_size)
                    predictions = logits[:-1].argmax(dim=-1)  # (chunk_size - 1,)
                    targets = chunk_ids[0, 1:]  # (chunk_size - 1,)

                    mask = answer_mask[:-1]
                    answer_positions = mask.nonzero(as_tuple=True)[0]

                    for pos in answer_positions:
                        total_answer_tokens += 1
                        if predictions[pos].item() == targets[pos].item():
                            correct_answer_tokens += 1

                    # Per-QA-pair exact match (approximate: check consecutive answer spans)
                    # For simplicity, check if ALL answer tokens are correct
                    if len(answer_positions) > 0:
                        all_correct = all(
                            predictions[p].item() == targets[p].item()
                            for p in answer_positions
                        )
                        total_pairs += 1
                        if all_correct:
                            exact_match_pairs += 1

                    # Debug output
                    if verbose and debug_count < n_debug:
                        debug_count += 1
                        print(f"\n  [doc {doc_idx}] gap={gap}, "
                              f"{len(answer_positions)} answer tokens")

                        # Show QA text and predictions
                        qa_tokens = chunk_ids[0].tolist()
                        qa_text = dataset.tokenizer.decode(qa_tokens, skip_special_tokens=True)
                        print(f"    QA: {qa_text[:200]}...")

                        # Show answer predictions
                        for i, pos in enumerate(answer_positions[:20]):
                            expected = dataset.tokenizer.decode([targets[pos].item()])
                            predicted = dataset.tokenizer.decode([predictions[pos].item()])
                            match = "Y" if predictions[pos].item() == targets[pos].item() else "N"
                            if i < 10:
                                print(f"    pos {pos.item()}: expected='{expected}' "
                                      f"predicted='{predicted}' [{match}]")

    token_accuracy = correct_answer_tokens / max(total_answer_tokens, 1)
    exact_match = exact_match_pairs / max(total_pairs, 1)
    attn_on_passage = (sum(attn_on_passage_list) / len(attn_on_passage_list)
                       if attn_on_passage_list else None)

    if verbose:
        print(f"\n  Token accuracy: {correct_answer_tokens}/{total_answer_tokens} "
              f"= {token_accuracy:.1%}")
        print(f"  Exact match: {exact_match_pairs}/{total_pairs} "
              f"= {exact_match:.1%}")
        if attn_on_passage is not None:
            print(f"  Attention on passage slots: {attn_on_passage:.1%}")

    # Gate analysis
    gate_analysis = _analyze_gate_values(model, dataset, device)
    if gate_analysis and verbose:
        print(f"\n  Gate values by chunk type:")
        for ctype in ["passage", "filler", "qa"]:
            g = gate_analysis[ctype]
            print(f"    {ctype.upper():>8s}: mean={g['mean']:.6f}, std={g['std']:.6f}")
        ratio = gate_analysis['passage']['mean'] / max(gate_analysis['filler']['mean'], 1e-10)
        print(f"    Passage/Filler ratio: {ratio:.1f}x")

    return {
        "token_accuracy": token_accuracy,
        "exact_match": exact_match,
        "correct_tokens": correct_answer_tokens,
        "total_tokens": total_answer_tokens,
        "exact_match_pairs": exact_match_pairs,
        "total_pairs": total_pairs,
        "gate_analysis": gate_analysis,
        "attn_on_passage": attn_on_passage,
    }


def _analyze_gate_values(model, dataset, device, n_docs=50):
    """Analyze gate values by chunk type."""
    from rpvt.model.cross_attention_memory import MemoryBank
    memory_modules = []
    for module in model.modules():
        if isinstance(module, HopfieldMemory) and module.write_mode == "gate":
            memory_modules.append(module)

    if not memory_modules:
        # Check for MemoryBank (cross-attention mode) — gate analysis via hook
        banks = [m for m in model.modules() if isinstance(m, MemoryBank)]
        if not banks:
            return None
        # For MemoryBank, do simplified gate analysis
        bank = banks[0]
        gate_by_type = {"passage": [], "filler": [], "qa": []}
        for doc in dataset.documents[:n_docs]:
            chunks = doc["chunks"]
            n_chunks = len(chunks)
            n_passage = doc["n_passage_chunks"]
            gap = doc["gap"]
            reset_memories(model)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_ids = chunk.unsqueeze(0).to(device)
                with torch.no_grad():
                    model(chunk_ids)
                # We can't easily capture gate values mid-forward for MemoryBank
                # Just return None for now
        return None

    gate_by_type = {"passage": [], "filler": [], "qa": []}
    mem = memory_modules[0]

    for doc in dataset.documents[:n_docs]:
        chunks = doc["chunks"]
        n_chunks = len(chunks)
        n_passage = doc["n_passage_chunks"]
        gap = doc["gap"]

        reset_memories(model)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_ids = chunk.unsqueeze(0).to(device)

            # Capture gate values
            mem._captured_gates = None
            orig_forward = mem.forward

            def make_capture_forward(m, orig):
                def capture_forward(x, chunk_size=64):
                    param_dtype = m.W_gate.weight.dtype
                    x_cast = x.to(dtype=param_dtype)
                    m._captured_gates = torch.sigmoid(m.W_gate(x_cast)).detach().cpu().squeeze(-1).squeeze(0)
                    return orig(x, chunk_size)
                return capture_forward

            mem.forward = make_capture_forward(mem, orig_forward)

            with torch.no_grad():
                model(chunk_ids)

            mem.forward = orig_forward

            if mem._captured_gates is not None:
                chunk_mean = mem._captured_gates.mean().item()
                if chunk_idx < n_passage:
                    gate_by_type["passage"].append(chunk_mean)
                elif chunk_idx < n_passage + gap:
                    gate_by_type["filler"].append(chunk_mean)
                else:
                    gate_by_type["qa"].append(chunk_mean)

    result = {}
    for ctype in ["passage", "filler", "qa"]:
        vals = gate_by_type[ctype]
        if vals:
            t = torch.tensor(vals)
            result[ctype] = {"mean": t.mean().item(), "std": t.std().item()}
        else:
            result[ctype] = {"mean": 0.0, "std": 0.0}
    return result


def main():
    parser = argparse.ArgumentParser(description="v3.2: NLP recall with pretrained model + Hopfield memory")

    # Model
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--memory-layer", type=int, default=-1,
                        help="Layer to attach memory (-1 = middle)")
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--n-slots", type=int, default=64)
    parser.add_argument("--decay", type=float, default=0.999)
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    parser.add_argument("--init-qk-shared", action="store_true")
    parser.add_argument("--n-extract", type=int, default=1,
                        help="Number of extraction queries per chunk (1=mean-pool, >1=learned selection)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-targets", type=str, default="q_proj,v_proj")

    # Ablations
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--memory-mode", type=str, default="additive",
                        choices=["additive", "cross_attn"],
                        help="Memory injection: 'additive' (residual) or 'cross_attn' (KV pairs in attention)")

    # Task
    parser.add_argument("--data-source", type=str, default="synthetic",
                        help="Data source: 'synthetic' (single), 'synthetic_multi' (two passages), "
                             "'synthetic_confusable' (two passages sharing attributes), "
                             "'synthetic_n_N' (N passages, e.g. synthetic_n_5), or 'squad'")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-qa-pairs", type=int, default=3)
    parser.add_argument("--gap-min", type=int, default=2)
    parser.add_argument("--gap-max", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=100)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr-memory", type=float, default=1e-3)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument("--retrieval-loss-weight", type=float, default=0.0,
                        help="Weight for auxiliary retrieval alignment loss")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/exp_v3_2")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build model
    model, tokenizer = build_model(
        model_name=args.model_name,
        device=args.device,
        memory_layer=args.memory_layer,
        memory_size=args.memory_size,
        n_slots=args.n_slots,
        decay=args.decay,
        gate_bias=args.gate_bias,
        lora_rank=args.lora_rank,
        lora_targets=args.lora_targets,
        no_memory=args.no_memory,
        no_lora=args.no_lora,
        init_qk_shared=args.init_qk_shared,
        n_extract=args.n_extract,
        memory_mode=args.memory_mode,
    )

    # Create datasets
    print(f"\nCreating datasets (seed={args.seed})...")
    train_data = SQuADRecallDataset(
        tokenizer=tokenizer, split="train",
        n_docs=args.n_train, chunk_size=args.chunk_size,
        gap_range=(args.gap_min, args.gap_max),
        max_qa_pairs=args.max_qa_pairs, seed=args.seed,
        data_source=args.data_source,
    )
    eval_data = SQuADRecallDataset(
        tokenizer=tokenizer, split="validation",
        n_docs=args.n_eval, chunk_size=args.chunk_size,
        gap_range=(args.gap_min, args.gap_max),
        max_qa_pairs=args.max_qa_pairs, seed=args.seed + 1000,
        data_source=args.data_source,
    )

    # Run no-memory baseline first
    if not args.no_memory:
        print(f"\n=== No-memory baseline (LoRA only, per-chunk) ===")
        baseline = evaluate(model, eval_data, device=args.device, verbose=False)
        print(f"  Baseline token accuracy: {baseline['token_accuracy']:.1%}")
        print(f"  Baseline exact match: {baseline['exact_match']:.1%}")

    # Train and evaluate
    results = train_and_eval(
        model, train_data, eval_data, args.device,
        num_epochs=args.epochs,
        lr_memory=args.lr_memory,
        lr_lora=args.lr_lora,
        log_every=args.log_every,
        two_phase=args.two_phase,
        output_dir=args.output_dir,
        retrieval_loss_weight=args.retrieval_loss_weight,
    )

    results["config"] = vars(args)

    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
