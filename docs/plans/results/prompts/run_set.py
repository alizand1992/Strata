"""Send the fixed prompt set (p*.json next to this file) to a running server, one after the other, and print each
prompt's tokens, prompt tok/s, decode tok/s, finish reason and the start of the reply (or the tool it called).
The speeds are the server's own `timings` (prompt_per_second, predicted_per_second).

    python docs/plans/results/prompts/run_set.py [--url http://127.0.0.1:8080] [--key-env STRATA_API_KEY] [--json out.json]

The key is read from the environment variable --key-env names (no key: none is sent)."""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def post(url, key, body, timeout=7200):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.time() - t0


def summary(name, out, wall):
    c = out["choices"][0]
    m = c["message"]
    t = out.get("timings") or {}
    u = out.get("usage") or {}
    tools = [x["function"]["name"] for x in m.get("tool_calls") or []]
    text = (m.get("content") or "").replace("\n", " ")[:120]
    return {"name": name, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "prompt_tok_s": t.get("prompt_per_second"), "decode_tok_s": t.get("predicted_per_second"),
            "wall_s": round(wall, 1), "finish": c.get("finish_reason"), "tools": tools, "start": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key-env", default="STRATA_API_KEY")
    ap.add_argument("--json", help="also write the rows to this file")
    a = ap.parse_args()
    key = os.environ.get(a.key_env)
    rows = []
    for f in sorted(HERE.glob("p*.json")):
        out, wall = post(a.url, key, json.loads(f.read_text()))
        r = summary(f.stem, out, wall)
        rows.append(r)
        print(f"{r['name']}: {r['prompt_tokens']} tokens, {r['prompt_tok_s']} / {r['decode_tok_s']} tok/s, "
              f"{r['wall_s']} s, {r['finish']}{' tools=' + ','.join(r['tools']) if r['tools'] else ''} :: {r['start']}",
              flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
