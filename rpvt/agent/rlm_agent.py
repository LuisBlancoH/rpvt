"""Recursive Language Model agent.

Instead of stuffing documents into context, the model writes Python code
to navigate, filter, and process them. It can recursively call itself
on subsets of the data.

Inspired by MIT's RLM paper (Zhang, Kraska, Khattab 2025).
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpvt.agent.file_readers import read_file


SYSTEM_PROMPT = """You solve tasks by writing Python code. Always use code — never guess or reason about data in your head.

Put code in ```python blocks — it runs and you see output. You can run multiple rounds.

When you have the final answer, say ANSWER: <value>. Give ONLY the value — no explanation.

Available variables:
- `context`: string with loaded file contents (if provided)
- `file_path`: path to the original file (if provided)
"""

FOLLOWUP_PROMPT = """Output:
```
{output}
```

{instruction}"""


# Template for the rlm() function injected into code
RLM_FUNCTION_TEMPLATE = '''
import subprocess, tempfile, json

def rlm(query, context_text=""):
    """Recursively call the AI model on a subset of data."""
    script = f"""
import sys
sys.path.insert(0, ".")
from rpvt.agent.rlm_agent import _rlm_subprocess_call
result = _rlm_subprocess_call(
    model_path="{model_path}",
    query=\\"\\"\\"{query}\\"\\"\\"  ,
    context_text=\\"\\"\\"{context_text[:2000]}\\"\\"\\"
)
print(result)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        try:
            r = subprocess.run([".venv/bin/python", f.name], capture_output=True, text=True, timeout=30)
            return r.stdout.strip()
        finally:
            import os; os.unlink(f.name)
'''


def execute_python(code, context=None, file_path=None, timeout=30):
    """Execute code with context and file_path available as variables."""
    preamble_parts = []

    # Context injection
    if context is not None:
        ctx_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        ctx_file.write(context)
        ctx_file.flush()
        ctx_file.close()
        preamble_parts.append(
            f'context = open("{ctx_file.name}", "r", errors="replace").read()'
        )
    else:
        preamble_parts.append('context = ""')

    # File path
    if file_path:
        preamble_parts.append(f'file_path = "{file_path}"')
    else:
        preamble_parts.append('file_path = None')

    preamble = "\n".join(preamble_parts) + "\n\n"
    full_code = preamble + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                [".venv/bin/python", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(__file__).parent.parent.parent),
            )
            output = result.stdout.strip()
            stderr = result.stderr.strip()
            if result.returncode != 0 and not output:
                # Show just the last useful error line
                err_lines = stderr.split("\n")
                useful = [l for l in err_lines if not l.startswith("  ")]
                output = f"ERROR: {useful[-1] if useful else stderr[-200:]}"
            if len(output) > 3000:
                output = output[:3000] + "\n... (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return f"ERROR: Code execution timed out ({timeout}s limit)"
        finally:
            os.unlink(f.name)
            if context is not None:
                try:
                    os.unlink(ctx_file.name)
                except OSError:
                    pass


def extract_code(text):
    """Extract Python code from response."""
    code = None
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.startswith("python\n"):
                code = code[7:]
            code = code.strip()

    if code:
        code = _auto_print_last_expr(code)
    return code


def _auto_print_last_expr(code):
    """If the last line is a bare expression (not assignment, not print, not
    control flow), wrap it in print() so it produces output like a notebook."""
    lines = code.rstrip().split("\n")
    last = lines[-1].strip()
    if not last or last.startswith("#"):
        return code
    # Already has print or is a statement
    skip_prefixes = (
        "print", "import ", "from ", "if ", "for ", "while ", "def ",
        "class ", "return ", "raise ", "try:", "except", "finally:",
        "with ", "assert ", "del ", "pass", "break", "continue",
        "elif ", "else:",
    )
    if last.startswith(skip_prefixes):
        return code
    # Assignment (but not ==)
    if "=" in last and "==" not in last and not last.startswith("("):
        # Check it's actually assignment, not augmented comparison
        for op in ["<=", ">=", "!="]:
            if op in last:
                break
        else:
            return code
    # It's a bare expression — wrap in print()
    lines[-1] = f"print({last})"
    return "\n".join(lines)


class RLMAgent:
    """Recursive Language Model agent with code-based reasoning."""

    def __init__(self, model_name="Qwen/Qwen3-8B", device="cuda",
                 max_steps=5, quantize=True):
        self.model_name = model_name
        self.device = device
        self.max_steps = max_steps

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        kwargs = dict(trust_remote_code=True, device_map="auto")
        if quantize:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **kwargs
        )

        # Warmup
        inputs = self.tokenizer("Hello", return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(
                **inputs, max_new_tokens=5,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM: {vram:.1f} GB, max_steps: {max_steps}")
        print(f"  Ready.")

    def _generate(self, messages, max_new_tokens=400, temperature=0.0):
        """Generate a response, streaming tokens to stdout."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Truncate if too long (keep end which has recent context)
        max_input = 6144
        if inputs.input_ids.shape[1] > max_input:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_input:]

        print("    > ", end="", flush=True)
        generated_ids = []
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values = None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)
            token_id = next_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(token_id)
            token_str = self.tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
            print(token_str, end="", flush=True)

            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype)
            ], dim=1)

        print(flush=True)
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

    def ask(self, question, file_content=None, file_path=None):
        """Ask a question using iterative code execution."""
        # Build task prompt
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

        # Iterative execution loop
        final_answer = None
        for step in range(self.max_steps):
            response = self._generate(messages, max_new_tokens=400)

            # Check if model gave a direct ANSWER:
            answer = _extract_answer(response)
            if answer is not None:
                final_answer = answer
                break

            code = extract_code(response)
            if code:
                output = execute_python(
                    code, context=file_content, file_path=file_path,
                )
                print(f"    [step {step+1}] → {output[:300]}")

                # Check if the code output contains ANSWER:
                answer = _extract_answer(output)
                if answer is not None:
                    final_answer = answer
                    break

                # On last step, take last line of output as answer
                if step == self.max_steps - 1:
                    if output and not output.startswith("ERROR:"):
                        lines = output.strip().split("\n")
                        final_answer = lines[-1].strip()
                    else:
                        final_answer = "UNKNOWN"
                    break

                messages.append({"role": "assistant", "content": response})

                remaining = self.max_steps - step - 1
                if output.startswith("ERROR:"):
                    instruction = f"Fix the error. {remaining} step(s) left."
                elif remaining == 1:
                    instruction = "Last step. Give your ANSWER:"
                else:
                    instruction = f"{remaining} step(s) left."

                messages.append({
                    "role": "user",
                    "content": FOLLOWUP_PROMPT.format(
                        output=output, instruction=instruction
                    ),
                })
            else:
                # No code, no ANSWER: — push to use code (unless last step)
                if step < self.max_steps - 1:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "Use Python code to solve this. Write a ```python block.",
                    })
                else:
                    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
                    if lines:
                        final_answer = lines[-1]
                    break

        if final_answer is None:
            final_answer = "UNKNOWN"

        return final_answer


    def solve_with_reflection(self, question, file_content=None,
                               file_path=None, max_attempts=2):
        """Solve with self-reflection loop.

        Attempt 1: normal solve.
        If we get an answer, ask the model to verify it with code.
        If verification fails or finds issues, retry with the reflection.
        """
        trajectories = []

        for attempt in range(max_attempts):
            if attempt == 0:
                answer = self.ask(question, file_content, file_path)
            else:
                # Retry with reflection context
                reflection_prompt = (
                    f"Previous attempt gave: {prev_answer}\n"
                    f"Reflection: {reflection}\n\n"
                    f"Try again from scratch. {question}"
                )
                answer = self.ask(reflection_prompt, file_content, file_path)

            print(f"    [attempt {attempt+1}] answer: {answer}")

            # Self-verification: ask model to check the answer with code
            verify_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Task: {question}\n"
                    f"Proposed answer: {answer}\n\n"
                    f"Write Python code to verify this answer is correct. "
                    f"Print VERIFIED if correct, or print WRONG: <reason> if not."
                    + (f"\nFile path: {file_path}" if file_path else "")
                )},
            ]
            verify_response = self._generate(verify_messages, max_new_tokens=300)
            code = extract_code(verify_response)

            if code:
                verify_output = execute_python(
                    code, context=file_content, file_path=file_path,
                )
                print(f"    [verify] → {verify_output[:200]}")

                if "VERIFIED" in verify_output.upper():
                    trajectories.append({
                        "attempt": attempt + 1,
                        "answer": answer,
                        "verified": True,
                    })
                    return answer, trajectories

                # Verification found a problem — reflect
                reflection = verify_output
            else:
                # Model couldn't write verification code
                reflection = verify_response

            prev_answer = answer
            trajectories.append({
                "attempt": attempt + 1,
                "answer": answer,
                "verified": False,
                "reflection": reflection[:500],
            })

        # Return last answer even if unverified
        return answer, trajectories


def _extract_answer(text):
    """Extract answer from text if it contains ANSWER: marker."""
    for line in text.split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("ANSWER:"):
            return line[7:].strip()
        if upper.startswith("FINAL ANSWER:"):
            return line[13:].strip()
    return None
