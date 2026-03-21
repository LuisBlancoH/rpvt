"""Claude-powered agent with code execution and failure learning.

Uses Claude Sonnet for reasoning, can execute Python code,
stores failures as lessons for future reference.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import anthropic

from rpvt.agent.file_readers import read_file


SYSTEM_PROMPT = """You are a precise assistant that answers questions based on provided documents and context.

You have access to a Python environment. You can write Python code in ```python blocks and it will be executed.

DECISION GUIDE - when to use code vs answer directly:
- Counting items, filtering data, math calculations → USE CODE
- Reading/parsing spreadsheets, PDFs, structured data → USE CODE
- Simple factual questions, logic, language tasks → ANSWER DIRECTLY
- Riddles, word puzzles, text manipulation → ANSWER DIRECTLY (think step by step)

When writing code:
- print() ONLY the final answer as the last line of output
- Use openpyxl for .xlsx, fitz for .pdf, python-docx for .docx
- Handle edge cases

When answering directly:
- Give ONLY the final answer, nothing else
- Be concise

If you truly cannot determine the answer, say "UNKNOWN"
"""

LEARNING_PROMPT = """You just attempted a task and got it wrong.

Task: {question}
Your answer: {predicted}
Correct answer: {gold}
{file_context}

Write a short, specific lesson learned that would help you get similar tasks right in the future.
Be specific and actionable. One or two sentences."""


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
                cwd=str(Path(__file__).parent.parent.parent),  # project root
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
    # Look for ```python blocks
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    # Look for ``` blocks
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.startswith("python\n"):
                code = code[7:]
            return code.strip()
    return None


class FailureMemory:
    """Stores lessons learned from failures."""

    def __init__(self, path="~/.rpvt/failure_memory.json"):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lessons = []
        if self.path.exists():
            with open(self.path) as f:
                self.lessons = json.load(f)

    def add(self, task_type, question, predicted, gold, lesson):
        entry = {
            "task_type": task_type,
            "question": question[:300],
            "predicted": str(predicted)[:200],
            "gold": str(gold),
            "lesson": lesson,
            "timestamp": time.time(),
        }
        self.lessons.append(entry)
        self._save()

    def retrieve(self, question, top_k=3):
        if not self.lessons:
            return []
        question_words = set(question.lower().split())
        scored = []
        for lesson in self.lessons:
            lesson_words = set(lesson["question"].lower().split())
            lesson_words.update(lesson["lesson"].lower().split())
            overlap = len(question_words & lesson_words)
            scored.append((overlap, lesson))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:top_k] if s[0] > 2]

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.lessons, f, indent=2)

    def __len__(self):
        return len(self.lessons)


class ClaudeAgent:
    """Agent powered by Claude Sonnet with code execution and failure learning."""

    def __init__(self, model="claude-sonnet-4-20250514",
                 failure_memory_path="~/.rpvt/failure_memory.json",
                 enable_code=True):
        self.client = anthropic.Anthropic()
        self.model = model
        self.memory = FailureMemory(failure_memory_path)
        self.enable_code = enable_code
        print(f"Claude Agent ready (model: {model})")
        print(f"  Code execution: {'enabled' if enable_code else 'disabled'}")
        print(f"  Failure memory: {len(self.memory)} lessons loaded")

    def ask(self, question, file_content=None, file_path=None,
            use_lessons=True, max_retries=1):
        """Ask a question with optional document context and code execution."""
        messages_content = []

        # Add relevant lessons
        if use_lessons:
            lessons = self.memory.retrieve(question)
            if lessons:
                lesson_text = "LESSONS FROM PAST MISTAKES (apply these):\n"
                for i, lesson in enumerate(lessons, 1):
                    lesson_text += f"{i}. {lesson['lesson']}\n"
                messages_content.append(lesson_text + "\n")

        # Add document context
        if file_content:
            max_chars = 50000
            if len(file_content) > max_chars:
                file_content = file_content[:max_chars] + "\n... (truncated)"
            messages_content.append(f"Document content:\n{file_content}\n")

        if file_path:
            messages_content.append(
                f"The document has also been saved to: {file_path}\n"
                "You can read it directly in your Python code.\n"
            )

        messages_content.append(question)

        if self.enable_code:
            messages_content.append(
                "\nYou can write Python code in ```python blocks if needed (it will be executed). "
                "Only use code for computation, counting, or data analysis. "
                "For simple questions, just answer directly. "
                "Give ONLY the final answer."
            )
        else:
            messages_content.append(
                "\nAnswer with ONLY the final answer, nothing else. Be concise."
            )

        user_message = "\n".join(messages_content)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text.strip()

        # Check if response contains code to execute
        if self.enable_code:
            code = extract_code(answer)
            if code:
                output = execute_python(code)
                if output and not output.startswith("ERROR:"):
                    # Return the last line of output as the answer
                    lines = output.strip().split("\n")
                    return lines[-1].strip()
                elif output.startswith("ERROR:") and max_retries > 0:
                    # Try to fix the code
                    fix_response = self.client.messages.create(
                        model=self.model,
                        max_tokens=1000,
                        system=SYSTEM_PROMPT,
                        messages=[
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": answer},
                            {"role": "user", "content":
                             f"The code produced an error:\n{output}\n\n"
                             "Fix the code and try again. Print ONLY the final answer."},
                        ],
                    )
                    fix_answer = fix_response.content[0].text.strip()
                    fix_code = extract_code(fix_answer)
                    if fix_code:
                        fix_output = execute_python(fix_code)
                        if fix_output and not fix_output.startswith("ERROR:"):
                            lines = fix_output.strip().split("\n")
                            return lines[-1].strip()
                    return output  # Return error if fix failed

        # No code — return the text answer (last line, stripped)
        lines = answer.strip().split("\n")
        return lines[-1].strip()

    def learn_from_failure(self, question, predicted, gold, file_content=None):
        """Generate and store a lesson from a failure."""
        file_context = ""
        if file_content:
            file_context = f"\nDocument snippet: {file_content[:1000]}"

        prompt = LEARNING_PROMPT.format(
            question=question[:500],
            predicted=predicted,
            gold=gold,
            file_context=file_context,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        lesson = response.content[0].text.strip()

        q_lower = question.lower()
        if any(w in q_lower for w in ["how many", "count", "number of"]):
            task_type = "counting"
        elif any(w in q_lower for w in ["calculate", "average", "sum", "deviation"]):
            task_type = "computation"
        elif any(w in q_lower for w in ["which", "what is the", "who"]):
            task_type = "lookup"
        elif any(w in q_lower for w in ["riddle", "puzzle", "logic"]):
            task_type = "reasoning"
        else:
            task_type = "general"

        self.memory.add(task_type, question, predicted, gold, lesson)
        return lesson

    def clear_memory(self):
        self.memory.lessons = []
        self.memory._save()
