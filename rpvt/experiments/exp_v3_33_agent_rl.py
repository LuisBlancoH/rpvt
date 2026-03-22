"""v3.33: Agent RL training with value-guided search.

Trains the agent with:
  - Supervised learning on L1/L2/L4 (QA + code demonstrations)
  - RL on L3 (interactive tasks with reward signal)
  - Value extraction query for advantage estimation
  - Memory persists across episodes (experience accumulates)

Usage:
    python -m rpvt.experiments.exp_v3_33_agent_rl
    python -m rpvt.experiments.exp_v3_33_agent_rl --checkpoint results/agent_3b/checkpoint_epoch10.pt
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.model.recurrent_memory import RecurrentMemoryTransformer
from rpvt.experiments.exp_v3_32_agent import (
    gen_l1, gen_l2, gen_person, gen_l3_training_sample,
    InteractiveEnv, CODE_TASKS, execute_code,
    eval_qa, eval_interactive, eval_code, run_eval,
    train_qa_batch, train_code_batch,
)


def rl_episode(model, tokenizer, device, rng, gamma=0.99,
               persistent_memory=True):
    """Run one RL episode on interactive task.

    Returns policy_loss, value_loss, reward, info dict.
    """
    if not persistent_memory:
        model.reset_memory()

    env = InteractiveEnv(rng)
    obs = env.reset()

    # Store task in memory
    obs_ids = tokenizer(obs, return_tensors="pt", truncation=True,
                        max_length=256).input_ids.to(device)
    model.forward(obs_ids, n_passes=1)

    steps = []
    done = False
    total_reward = 0

    for step_idx in range(5):
        # Generate action with exploration
        prompt = "Action:"
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Forward to get logits + value
        logits, _, info = model.forward(prompt_ids, return_info=True)
        value = info["value"]

        # Sample action tokens (with temperature for exploration)
        next_logits = logits[:, -1, :] / 0.7  # temperature
        probs = torch.softmax(next_logits, dim=-1)

        # Generate multiple tokens for the action
        action_tokens = []
        action_log_probs = []
        current_ids = prompt_ids

        for _ in range(30):  # max action length
            with torch.no_grad():
                out_logits, _ = model.forward(current_ids, n_passes=1)
            next_logits = out_logits[:, -1, :] / 0.7

            probs = torch.softmax(next_logits, dim=-1)
            token = torch.multinomial(probs, 1)
            log_prob = torch.log(probs[0, token.item()] + 1e-8)

            action_tokens.append(token.item())
            action_log_probs.append(log_prob)

            current_ids = token.unsqueeze(0)

            # Stop at newline or EOS
            decoded = tokenizer.decode([token.item()])
            if "\n" in decoded or token.item() == tokenizer.eos_token_id:
                break

        action_text = tokenizer.decode(action_tokens, skip_special_tokens=True)
        action_text = action_text.split("\n")[0].strip()

        # Execute in environment
        result, reward, done = env.step(action_text)
        total_reward = max(total_reward, reward)

        # Store experience in memory (just the result + reward, no forced reflection)
        exp_text = f"Action: {action_text}\nResult: {result}"
        exp_ids = tokenizer(exp_text, return_tensors="pt", truncation=True,
                            max_length=128).input_ids.to(device)
        model.forward(exp_ids, n_passes=1)

        # Store step data for RL update
        mean_log_prob = torch.stack(action_log_probs).mean()
        steps.append({
            "log_prob": mean_log_prob,
            "value": torch.tensor(value, device=device, dtype=torch.float32),
            "reward": reward,
        })

        if done:
            break

    if not steps:
        return None, None, 0, {}

    # Compute TD advantages
    policy_loss = torch.tensor(0.0, device=device)
    value_loss = torch.tensor(0.0, device=device)

    for t in range(len(steps)):
        # TD target
        if t == len(steps) - 1:
            next_value = 0.0  # terminal
        else:
            next_value = steps[t + 1]["value"].item()

        td_target = steps[t]["reward"] + gamma * next_value
        advantage = td_target - steps[t]["value"].item()

        # Policy loss: reinforce good actions, suppress bad
        policy_loss = policy_loss - advantage * steps[t]["log_prob"]

        # Value loss: train value prediction
        value_loss = value_loss + (steps[t]["value"] - td_target) ** 2

    n_steps = len(steps)
    policy_loss = policy_loss / n_steps
    value_loss = value_loss / n_steps

    return policy_loss, value_loss, total_reward, {
        "n_steps": n_steps,
        "action_texts": [action_text],
        "final_reward": total_reward,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-supervised", type=int, default=600,
                        help="Supervised steps per epoch")
    parser.add_argument("--n-rl", type=int, default=200,
                        help="RL episodes per epoch")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--persistent-memory", action="store_true", default=True,
                        help="Keep memory across RL episodes")
    parser.add_argument("--output-dir", type=str, default="results/agent_rl")
    args = parser.parse_args()

    device = "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True)
    model = RecurrentMemoryTransformer(
        qwen, n_memory_tokens=16, lora_rank=args.lora_rank, max_passes=3)
    model = model.to(device)
    del qwen
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Initial eval
    initial = run_eval(model, tokenizer, device, "Initial")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    history = []

    print(f"\n--- Training ({args.epochs} epochs: {args.n_supervised} supervised "
          f"+ {args.n_rl} RL per epoch) ---")

    for epoch in range(args.epochs):
        model.train()
        rng = random.Random(epoch + 42)
        t0 = time.time()

        # === Phase 1: Supervised (L1 + L2 + L3 demos + L4 code) ===
        total_sup_loss = 0
        n_sup = 0
        optimizer.zero_grad()

        for step in range(args.n_supervised):
            r = rng.random()
            if r < 0.35:
                loss = train_qa_batch(model, tokenizer, device, rng, "L1")
            elif r < 0.60:
                loss = train_qa_batch(model, tokenizer, device, rng, "L2")
            elif r < 0.80:
                loss = train_qa_batch(model, tokenizer, device, rng, "L3")
            else:
                loss = train_code_batch(model, tokenizer, device, rng)

            if loss is not None and not torch.isnan(loss):
                (loss / args.grad_accum).backward()
                total_sup_loss += loss.item()
                n_sup += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 200 == 0:
                print(f"    [Supervised {step+1}/{args.n_supervised}] "
                      f"loss={total_sup_loss/max(n_sup,1):.4f}")

        # Flush
        if args.n_supervised % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_sup_loss = total_sup_loss / max(n_sup, 1)

        # === Phase 2: RL on interactive tasks ===
        total_policy_loss = 0
        total_value_loss = 0
        total_reward = 0
        n_rl = 0
        n_completed = 0

        # Reset memory for RL phase (fresh start for experience accumulation)
        model.reset_memory()

        for ep in range(args.n_rl):
            policy_loss, value_loss, reward, info = rl_episode(
                model, tokenizer, device, rng,
                persistent_memory=args.persistent_memory,
            )

            if policy_loss is not None:
                rl_loss = policy_loss + 0.5 * value_loss
                (rl_loss / args.grad_accum).backward()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_reward += reward
                n_rl += 1
                if reward > 0.5:
                    n_completed += 1

            if (ep + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (ep + 1) % 50 == 0:
                avg_rew = total_reward / max(n_rl, 1)
                comp_rate = n_completed / max(n_rl, 1)
                print(f"    [RL {ep+1}/{args.n_rl}] "
                      f"policy={total_policy_loss/max(n_rl,1):.4f} "
                      f"value={total_value_loss/max(n_rl,1):.4f} "
                      f"reward={avg_rew:.3f} "
                      f"completed={comp_rate:.1%} "
                      f"mem={model.memory_buffer.n_stored}/{model.memory_buffer.max_entries}")

        # Flush
        if args.n_rl % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - t0
        avg_rl_reward = total_reward / max(n_rl, 1)
        completion_rate = n_completed / max(n_rl, 1)

        # Eval
        eval_results = run_eval(model, tokenizer, device, f"Epoch {epoch+1}")

        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"sup_loss={avg_sup_loss:.4f}, "
              f"rl_reward={avg_rl_reward:.3f}, "
              f"rl_completed={completion_rate:.1%}, "
              f"({elapsed:.0f}s)")

        history.append({
            "epoch": epoch + 1,
            "sup_loss": avg_sup_loss,
            "rl_policy_loss": total_policy_loss / max(n_rl, 1),
            "rl_value_loss": total_value_loss / max(n_rl, 1),
            "rl_reward": avg_rl_reward,
            "rl_completion_rate": completion_rate,
            "eval": eval_results,
        })

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "history": history,
        }, output_dir / f"checkpoint_epoch{epoch+1}.pt")

    # Final
    final = run_eval(model, tokenizer, device, "Final")
    with open(output_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "history": history, "final": final},
                  f, indent=2)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
