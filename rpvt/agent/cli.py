"""Interactive CLI for the RPVT Document QA Agent.

Usage:
    python -m rpvt.agent.cli
    python -m rpvt.agent.cli --read doc1.txt doc2.txt
    python -m rpvt.agent.cli --max-entries 4096
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def format_status(status):
    """Format memory status for display."""
    pct = status["utilization"] * 100
    lines = [
        f"Memory: {status['n_stored']}/{status['max_entries']} entries ({pct:.1f}%)",
        f"Documents: {status['n_documents']}",
    ]
    for doc_id, meta in status["documents"].items():
        lines.append(f"  [{doc_id}] {meta['n_tokens']} tokens, "
                      f"{meta['n_chunks']} chunks, {meta['n_stored']} stored")
    return "\n".join(lines)


def print_help():
    print("""
RPVT Document QA Agent
======================
Commands:
  <text>              Ask a question (uses memory)
  /read <filepath>    Load a document into memory
  /paste              Enter text directly (end with Ctrl-D or empty line)
  /status             Show memory utilization
  /docs               List loaded documents
  /clear              Clear all memory
  /chat <message>     Chat without memory (baseline)
  /save [name]        Save session to disk
  /load <name>        Load a saved session
  /sessions           List saved sessions
  /help               Show this help
  /quit               Exit
""")


def get_session_dir():
    d = Path.home() / ".rpvt" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(agent, name=None):
    """Save agent state to disk."""
    if name is None:
        name = f"session_{int(time.time())}"
    session_dir = get_session_dir() / name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save KV memory tensors
    torch.save(agent.kv_memory.state_dict(), session_dir / "kv_memory.pt")

    # Save document metadata
    meta = {
        "model_name": agent.model_name,
        "max_entries": agent.kv_memory.max_entries,
        "chunk_size": agent.chunk_size,
        "documents": agent.documents,
    }
    with open(session_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return name, session_dir


def load_session(agent, name):
    """Load agent state from disk."""
    session_dir = get_session_dir() / name
    if not session_dir.exists():
        return f"Session not found: {name}"

    # Load metadata
    with open(session_dir / "metadata.json") as f:
        meta = json.load(f)

    if meta["model_name"] != agent.model_name:
        return (f"Model mismatch: session uses {meta['model_name']}, "
                f"agent uses {agent.model_name}")

    # Load KV memory
    state = torch.load(
        session_dir / "kv_memory.pt", map_location=agent.device, weights_only=True
    )
    agent.kv_memory.load_state_dict(state)
    agent.documents = meta.get("documents", {})

    return None  # success


def list_sessions():
    """List saved sessions."""
    session_dir = get_session_dir()
    sessions = []
    for d in sorted(session_dir.iterdir()):
        if d.is_dir() and (d / "metadata.json").exists():
            with open(d / "metadata.json") as f:
                meta = json.load(f)
            sessions.append({
                "name": d.name,
                "model": meta.get("model_name", "?"),
                "n_docs": len(meta.get("documents", {})),
            })
    return sessions


def handle_command(user_input, agent):
    """Parse and execute a command. Returns response string."""
    stripped = user_input.strip()
    if not stripped:
        return None

    # Commands
    if stripped.startswith("/"):
        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            print_help()
            return None

        elif cmd == "/quit" or cmd == "/exit":
            print("Goodbye.")
            sys.exit(0)

        elif cmd == "/read":
            if not arg:
                return "Usage: /read <filepath>"
            result = agent.ingest_file(arg)
            if "error" in result:
                return f"Error: {result['error']}"
            return (f"Loaded [{result['doc_id']}]: "
                    f"{result['n_tokens']} tokens, "
                    f"{result['n_stored']} entries stored "
                    f"({result['time']:.1f}s)")

        elif cmd == "/paste":
            print("Enter text (end with empty line or Ctrl-D):")
            lines = []
            try:
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
            except EOFError:
                pass
            if not lines:
                return "No text entered."
            text = "\n".join(lines)
            result = agent.ingest_text(text)
            if "error" in result:
                return f"Error: {result['error']}"
            return (f"Stored [{result['doc_id']}]: "
                    f"{result['n_tokens']} tokens, "
                    f"{result['n_stored']} entries stored")

        elif cmd == "/status":
            return format_status(agent.memory_status())

        elif cmd == "/docs":
            status = agent.memory_status()
            if not status["documents"]:
                return "No documents loaded."
            lines = []
            for doc_id, meta in status["documents"].items():
                lines.append(f"  [{doc_id}] {meta['n_tokens']} tokens, "
                             f"{meta['n_stored']} stored")
            return "\n".join(lines)

        elif cmd == "/clear":
            agent.reset_memory()
            return "Memory cleared."

        elif cmd == "/chat":
            if not arg:
                return "Usage: /chat <message>"
            return agent.generate(arg, use_memory=False)

        elif cmd == "/save":
            name = arg.strip() if arg.strip() else None
            name, path = save_session(agent, name)
            return f"Session saved: {name} ({path})"

        elif cmd == "/load":
            if not arg:
                return "Usage: /load <session_name>"
            err = load_session(agent, arg.strip())
            if err:
                return f"Error: {err}"
            status = agent.memory_status()
            return (f"Session loaded: {status['n_stored']} entries, "
                    f"{status['n_documents']} documents")

        elif cmd == "/sessions":
            sessions = list_sessions()
            if not sessions:
                return "No saved sessions."
            lines = []
            for s in sessions:
                lines.append(f"  {s['name']} ({s['model']}, {s['n_docs']} docs)")
            return "\n".join(lines)

        else:
            return f"Unknown command: {cmd}. Type /help for commands."

    # Default: ask a question using memory
    n_stored = agent.kv_memory.n_stored.item()
    if n_stored == 0:
        print("(No documents loaded — answering from model knowledge only)")
    return agent.generate(stripped, use_memory=True)


def main():
    parser = argparse.ArgumentParser(description="RPVT Document QA Agent")
    parser.add_argument("--model-name", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-entries", type=int, default=2048,
                        help="Max KV cache entries to store")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--read", type=str, nargs="*",
                        help="Files to pre-load into memory")
    parser.add_argument("--load-session", type=str, default=None,
                        help="Session to restore")

    args = parser.parse_args()

    from rpvt.agent.core import AgentCore

    agent = AgentCore(
        model_name=args.model_name,
        device=args.device,
        max_entries=args.max_entries,
        chunk_size=args.chunk_size,
    )

    # Load session if specified
    if args.load_session:
        err = load_session(agent, args.load_session)
        if err:
            print(f"Warning: {err}")
        else:
            status = agent.memory_status()
            print(f"Session restored: {status['n_stored']} entries, "
                  f"{status['n_documents']} documents")

    # Pre-load files if specified
    if args.read:
        for filepath in args.read:
            result = agent.ingest_file(filepath)
            if "error" in result:
                print(f"  Error loading {filepath}: {result['error']}")
            else:
                print(f"  Loaded [{result['doc_id']}]: "
                      f"{result['n_tokens']} tokens ({result['time']:.1f}s)")

    print("\nRPVT Agent ready. Type /help for commands.\n")

    # REPL
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        response = handle_command(user_input, agent)
        if response is not None:
            print(response)
            print()


if __name__ == "__main__":
    main()
