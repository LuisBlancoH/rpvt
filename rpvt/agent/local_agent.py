"""Local agent powered by Qwen3.5-9B with code execution.

Uses quantized local model for reasoning + Python code execution.
No API costs.
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


SYSTEM_PROMPT = """You are a precise assistant that answers questions based on provided documents and context.

You have access to a Python environment. You can write Python code in ```python blocks and it will be executed.

DECISION GUIDE - when to use code vs answer directly:
- Counting items, filtering data, math calculations → USE CODE
- Reading/parsing spreadsheets, PDFs, structured data → USE CODE
- Simple factual questions, logic, language tasks → ANSWER DIRECTLY
- Riddles, word puzzles, text manipulation → ANSWER DIRECTLY

When writing code:
- print() ONLY the final answer as the last line of output
- Use openpyxl for .xlsx, fitz for .pdf, python-docx for .docx

When answering directly:
- Give ONLY the final answer, nothing else

If you truly cannot determine the answer, say "UNKNOWN"
"""


def execute_python(code, timeout=30):
    """Execute Python code in a subprocess and return output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
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
            if result.returncode != 0 and not output:
                output = f"ERROR: {result.stderr.strip()[-500:]}"
            return output
        except subprocess.TimeoutExpired:
            return "ERROR: Code execution timed out"
        finally:
            os.unlink(f.name)


def extract_code(text):
    """Extract Python code from a response."""
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.startswith("python\n"):
                code = code[7:]
            return code.strip()
    return None


class LocalAgent:
    """Agent powered by local quantized model with code execution."""

    def __init__(self, model_name="Qwen/Qwen3.5-4B", device="cuda",
                 enable_code=True, quantize=False):
        self.model_name = model_name
        self.device = device
        self.enable_code = enable_code

        quant_label = "int4" if quantize else "bf16"
        print(f"Loading {model_name} ({quant_label})...")
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

        vram = torch.cuda.memory_allocated() / 1e9
        config = self.model.config
        print(f"  {config.hidden_size}d, {config.num_hidden_layers}L, "
              f"VRAM: {vram:.1f} GB")
        print(f"  Code execution: {'enabled' if enable_code else 'disabled'}")
        print(f"  Ready.")

    def ask(self, question, file_content=None, file_path=None,
            max_new_tokens=500):
        """Ask a question with optional document context and code execution."""
        parts = []

        # Add document context
        if file_content:
            max_chars = 12000  # Less than API model, limited context
            if len(file_content) > max_chars:
                file_content = file_content[:max_chars] + "\n... (truncated)"
            parts.append(f"Document content:\n{file_content}\n")

        if file_path:
            parts.append(f"The document is saved at: {file_path}\n"
                         "You can read it in Python code.\n")

        parts.append(question)

        if self.enable_code:
            parts.append(
                "\nYou can write Python code in ```python blocks if needed "
                "(it will be executed). Only use code for computation, counting, "
                "or data analysis. For simple questions, just answer directly. "
                "Give ONLY the final answer."
            )
        else:
            parts.append("\nAnswer with ONLY the final answer, nothing else.")

        user_content = "\n".join(parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        answer = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        # Check if response contains code to execute
        if self.enable_code:
            code = extract_code(answer)
            if code:
                output = execute_python(code)
                if output and not output.startswith("ERROR:"):
                    lines = output.strip().split("\n")
                    return lines[-1].strip()
                elif output.startswith("ERROR:"):
                    # Try once more with error feedback
                    retry_msg = (f"The code produced an error:\n{output}\n\n"
                                 "Fix the code. Print ONLY the final answer.")
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user", "content": retry_msg})

                    prompt2 = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    inputs2 = self.tokenizer(
                        prompt2, return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        out2 = self.model.generate(
                            **inputs2,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    answer2 = self.tokenizer.decode(
                        out2[0][inputs2.input_ids.shape[1]:],
                        skip_special_tokens=True
                    ).strip()

                    code2 = extract_code(answer2)
                    if code2:
                        output2 = execute_python(code2)
                        if output2 and not output2.startswith("ERROR:"):
                            lines = output2.strip().split("\n")
                            return lines[-1].strip()
                    return output  # Return original error

        # No code — return last line of answer
        lines = answer.strip().split("\n")
        return lines[-1].strip()
