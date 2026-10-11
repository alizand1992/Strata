"""The fixed prompt set of docs/plans/OPTION_A_EXCLUSIVE_RAM.md (Phase 0 onward): five OpenAI chat request bodies,
greedy (temperature 0) with a fixed max_tokens, so every phase sends the same bytes. The two code-reading prompts quote
source files at commit fb58e0d (read with `git show`, so a later edit of those files does not change the prompts).
Run from the repository root: python docs/plans/results/prompts/make_prompts.py"""
import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
COMMIT = "fb58e0d"


def src(path, max_bytes=None):
    text = subprocess.run(["git", "show", f"{COMMIT}:{path}"], check=True, capture_output=True, text=True).stdout
    if max_bytes and len(text) > max_bytes:
        text = text[:text.rfind("\n", 0, max_bytes) + 1]
    return text


def body(messages, max_tokens, tools=None):
    b = {"model": "strata", "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False}
    if tools:
        b["tools"] = tools
    return b


prompts = {
    "p1-code-4k": body([{"role": "user", "content":
        "Read this C++ file from an inference engine and explain, step by step, what RemoteExperts::begin and "
        "RemoteExperts::finish do, which entries a helper GPU takes, and what happens on failure.\n\n"
        "```cpp\n" + src("src/core/remote_experts.cpp") + "```"}], 512),
    "p2-code-32k": body([{"role": "user", "content":
        "Read the start of this C++ file from an inference engine. Summarise how the expert pool decides, per routed "
        "entry, whether an expert is computed on the GPU, over PCIe, or on the CPU, and list the functions involved.\n\n"
        "```cpp\n" + src("src/core/expert_source.cpp", 112_000) + "```"}], 512),
    "p3-chat-short": body([{"role": "user", "content":
        "What is the difference between a mutex and a semaphore? Answer in a few short paragraphs."}], 512),
    "p4-chat-short": body([{"role": "user", "content":
        "Write a Python function that returns the n-th Fibonacci number in O(log n) time, with a short explanation."}],
        512),
    "p5-tool-use": body([
        {"role": "system", "content": "You are a coding assistant. Use the tools when they help."},
        {"role": "user", "content": "Find where the function make_cache_complement_plan is defined in the repository "
                                    "and show me its first 40 lines."}], 256, tools=[
        {"type": "function", "function": {"name": "grep", "description": "Search files for a regex.",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"},
                                                            "path": {"type": "string"}}, "required": ["pattern"]}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read lines of a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                            "offset": {"type": "integer"},
                                                            "limit": {"type": "integer"}}, "required": ["path"]}}}]),
}

for name, b in prompts.items():
    (OUT / f"{name}.json").write_text(json.dumps(b, indent=1) + "\n", encoding="utf-8")
    print(name, len(json.dumps(b)), "bytes")
