"""An agent-style session against a running server: the repository's source as the first message (about --tokens
tokens), then --turns follow-up questions that each reuse the context (the earlier messages and replies are sent again,
as an agent does) and bring ~1,300 characters of further source, as a tool result would: about 400-550 new
tokens a turn with the reply. Greedy, --max-tokens a reply, a thinking budget of --think so a reply has text. Prints each turn's prompt tokens, prompt and
decode tok/s (the server's `timings`) and wall time, then the follow-ups' range and total.

The source is read at commit fb58e0d with `git show` (as make_prompts.py does), so the first message is the same
bytes on every run whatever the checkout. Run from the repository root:

    python docs/plans/results/prompts/agent_replay.py --tokens 20000 --turns 6
    python docs/plans/results/prompts/agent_replay.py --tokens 87000 --turns 4
    python docs/plans/results/prompts/agent_replay.py --tokens 195000 --turns 4     (a fresh server)

Rebuilt 2026-10-27: the first version (results 09 and 10) was never committed, so its first messages are not
these bytes. Compare replay numbers only between runs of this script."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_set import post, summary  # noqa: E402

COMMIT = "fb58e0d"
DIRS = ("src/core/", "src/prefill/", "src/program/", "serve/", "include/strata/")
EXTS = (".cpp", ".hpp", ".cu", ".py", ".h")
CHARS_PER_TOKEN = 3.4   # this tokenizer on this repo's source, about; the real count is the server's usage

FOLLOW_UPS = [
    "Which function in the files above decides how many experts go to the GPU cache? Name the file and say in three "
    "sentences how it decides. Then list two inputs that change its answer.",
    "Suppose the helper GPU stops answering in the middle of a reply. Walk through what the code above does, in order, "
    "naming the functions involved. Keep it under 200 words.",
    "Write a short unit test (pseudocode is fine) for the part of the server that builds the engine's command line "
    "from the config file. Say which config keys it should cover and why.",
    "Find one place in the code above where an error is reported to the user and suggest a clearer message. Quote the "
    "current text, give the new text, and explain the change in one sentence.",
    "List the environment variables the server code above reads, with one line each on what it changes. Only the "
    "first ten you find.",
    "Summarize, in five bullet points, how a chat request travels from the HTTP handler to the engine and back, "
    "naming the functions in order.",
]


def source(tokens, extra=0):
    """The first message's source and `extra` characters after it (the follow-ups' tool results)."""
    budget = int(tokens * CHARS_PER_TOKEN) + extra
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", COMMIT], check=True, capture_output=True,
                           text=True).stdout.split("\n")
    files = sorted(n for n in names if n.startswith(DIRS) and n.endswith(EXTS) and "/test" not in n)
    parts, size = [], 0
    for n in files:
        text = subprocess.run(["git", "show", f"{COMMIT}:{n}"], check=True, capture_output=True, text=True).stdout
        part = f"===== {n} =====\n{text}\n"
        if size + len(part) > budget:
            part = part[:budget - size]
            part = part[:part.rfind("\n") + 1]
        parts.append(part)
        size += len(part)
        if size >= budget:
            break
    text = "".join(parts)
    head = budget - extra
    head = text.rfind("\n", 0, head) + 1
    return text[:head], text[head:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key-env", default="STRATA_API_KEY")
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--think", type=int, default=128, help="reasoning_budget_tokens of each request")
    ap.add_argument("--json", help="also write the rows to this file")
    a = ap.parse_args()
    if a.turns > len(FOLLOW_UPS):
        ap.error(f"at most {len(FOLLOW_UPS)} turns")
    key = os.environ.get(a.key_env)
    TOOL = 1300
    first, more = source(a.tokens, TOOL * a.turns)
    messages = [{"role": "user", "content": "Here is the source of an inference server. Read it; I will ask "
                 "questions about it.\n\n" + first + "\nFirst: in one paragraph, what does this code do?"}]
    rows = []
    for i in range(a.turns + 1):
        if i > 0:
            tool = more[(i - 1) * TOOL:i * TOOL]
            messages.append({"role": "user", "content": FOLLOW_UPS[i - 1] + "\n\nThe next part of the source:\n" + tool})
        body = {"model": "strata", "messages": messages, "temperature": 0, "max_tokens": a.max_tokens, "stream": False,
                "reasoning_budget_tokens": a.think}
        out, wall = post(a.url, key, body)
        r = summary("first" if i == 0 else f"turn {i}", out, wall)
        rows.append(r)
        print(f"{r['name']}: {r['prompt_tokens']} tokens, {r['prompt_tok_s']} / {r['decode_tok_s']} tok/s, "
              f"{r['wall_s']} s, {r['finish']} :: {r['start']}", flush=True)
        messages.append({"role": "assistant", "content": out["choices"][0]["message"].get("content") or ""})
    turns = rows[1:]
    if turns:
        def rng(k):
            v = [r[k] for r in turns if r[k] is not None]
            return f"{min(v)}-{max(v)}" if v else "-"
        print(f"follow-ups: prompt tok/s {rng('prompt_tok_s')}, decode tok/s {rng('decode_tok_s')}, "
              f"wall {sum(r['wall_s'] for r in turns):.1f} s total")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
