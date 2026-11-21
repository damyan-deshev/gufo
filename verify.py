"""Verify the recorded HTTP evidence and rebuild the throughput tables offline."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from replay import phase_values, read_stream


ROOT = Path(__file__).resolve().parent
ARMS = ("old-off", "new-off", "new-k10")
SEEDS = (42, 43, 44)
GATE_ORDER = ("review-before", "candidate-default", "candidate-permitted", "review-after")


def read(path):
    return json.loads(path.read_text())


def check_run(record):
    folder = ROOT / record["case"]
    request = read(folder / "request.json")
    answer = read(folder / "result.json")
    stream = read_stream(gzip.decompress((folder / "response.sse.gz").read_bytes()), request)
    for key in ("usage", "finish", "scored_tokens", "content_bytes", "content_hash", "reasoning_bytes",
                "reasoning_hash", "wire_bytes", "sse_events"):
        assert stream[key] == answer[key], (folder, key)
    if "prompt" in request:
        assert stream["timings"] == answer["timings"]
    usage = stream["usage"]
    prompt, prefill_ms, generated, decode_ms = phase_values(stream)
    assert generated == answer["generated"] == int(record["generated"]) == request["max_tokens"] == 1024
    assert usage["prompt_tokens"] == answer["prompt"] == int(record["prompt"])
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0 and request["cache_prompt"] is False
    assert usage["total_tokens"] <= 16384 and int(record["seed"]) == request["seed"] == answer["seed"]
    assert request["model"] == record["model"] == answer["model"]
    assert answer["arm"] == record["arm"] and record["repeat"] == str(answer["repeat"])
    assert all(request[k] == v for k, v in answer["sampling"].items())
    assert all(usage.get(k, 0) == answer[v] for k, v in (("draft_tokens", "draft"), ("draft_tokens_accepted", "accepted")))
    if "gufo" in usage:
        assert usage["gufo"]["requested_logical_concurrency"] == usage["gufo"]["physical_execution_width"] == 1
    rates = (("prefill", prompt * 1000 / prefill_ms, "prompt_tokens_per_second"),
             ("decode", generated * 1000 / decode_ms, "completion_tokens_per_second"))
    for field, rate, wire_field in rates:
        for observed in (answer[field], float(record[field]), usage[wire_field]):
            assert math.isclose(observed, rate, rel_tol=1e-5), (folder, field)
    assert math.isclose(float(record["wall"]), answer["wall"], abs_tol=1e-7)
    for field in ("scored_tokens", "wire_bytes", "draft", "accepted"):
        assert int(record[field]) == answer[field], (folder, field)
    selector = {k: request[k] for k in ("logprobs", "top_logprobs") if k in request}
    if "prompt" in request:
        expected = {"new-off": {}, "new-k0": {"logprobs": 0}, "new-k10": {"logprobs": 10}}
    else:
        expected = {"old-off": {}, "new-off": {}, "new-k0": {"logprobs": True},
                    "new-k10": {"logprobs": True, "top_logprobs": 10}}
    assert selector == expected[record["arm"]]
    answer["paired_request"] = {k: v for k, v in request.items() if k not in selector}
    answer["phase_ms"] = (prefill_ms, decode_ms, answer["wall"] * 1000)
    return answer


def changed(new, old):
    return 100 * (new / old - 1)


def same_output(a, b):
    return all(a[k] == b[k] for k in ("generated", "finish", "content_hash", "reasoning_hash", "draft", "accepted"))


def trios(runs):
    normal = {(r["seed"], r["arm"]): r for r in runs if not r["repeat"]}
    assert len(normal) == 9 and len(runs) == 10
    repeat, = (r for r in runs if r["repeat"])
    assert repeat["arm"] == "new-off" and repeat["paired_request"] == normal[43, "new-off"]["paired_request"]
    for seed in SEEDS:
        old, off, top = (normal[seed, arm] for arm in ARMS)
        assert old["paired_request"] == off["paired_request"] == top["paired_request"]
        yield old, off, top


def render(primary, controls, gate, models, per_seed):
    print("| Model | Prefill tok/s, old / off / top 10 | Decode tok/s, old / off / top 10 | Off vs old | Top 10 vs off | Wall time, top 10 vs off |")
    print("|---|---:|---:|---:|---:|---:|")
    phase_flags = {"off": [0, 0, 0], "top": [0, 0, 0]}
    matches = Counter()
    seeds = []
    for mid, profile in models.items():
        triples = list(trios([r for r in primary if r["model"] == mid]))
        def rates(field):
            return " / ".join(f"{median(row[i][field] for row in triples):.1f}" for i in range(3))
        def delta(field, current, base):
            return median(changed(row[current][field], row[base][field]) for row in triples)
        print(f"| {profile['name']} | {rates('prefill')} | {rates('decode')} | {delta('decode',1,0):+.2f}% | {delta('decode',2,1):+.1f}% | {delta('wall',2,1):+.1f}% |")
        for old, off, top in triples:
            for label, a, b in (("off", old, off), ("top", off, top)):
                matches[label] += same_output(a, b)
                for i, (ref, tested) in enumerate(zip(a["phase_ms"], b["phase_ms"])):
                    phase_flags[label][i] += tested - ref > max(3, .05 * ref)
            seeds.append((profile["name"], old, off, top))
    print(f"\nSlowdowns past max(3 ms, 5%) out of 21 pairs (prefill / decode / wall): {phase_flags}")
    print(f"Same output, finish and draft counters: {sum(matches.values())}/42")
    print("\nSeed 43 follow-up, decode tok/s (off / chosen only / top 10)")
    for mid, profile in models.items():
        for raw in (False, True):
            rows = {r["arm"]: r for r in controls if r["model"] == mid and ("prompt" in r["paired_request"]) == raw}
            if not rows:
                continue
            a, b, c = (rows[arm] for arm in ("new-off", "new-k0", "new-k10"))
            assert a["paired_request"] == b["paired_request"] == c["paired_request"] and a["seed"] == 43
            print(f"{profile['name']}{' (raw)' if raw else ''}: {a['decode']:.2f} / {b['decode']:.2f} / {c['decode']:.2f}")
    print("\nQwen 35B startup check, prefill / decode tok/s / wall seconds")
    before = gate["review-before"]
    for label in GATE_ORDER:
        r = gate[label]
        assert r["paired_request"] == before["paired_request"] and same_output(r, before)
        print(f"{label}: {r['prefill']:.2f} / {r['decode']:.2f} / {r['wall']:.2f}")
    if per_seed:
        print("\n| Model | Seed | Prefill tok/s | Decode tok/s | Wall seconds |")
        print("|---|---:|---:|---:|---:|")
        for name, *rows in seeds:
            cells = [" / ".join(f"{r[field]:.{places}f}" for r in rows)
                     for field, places in (("prefill", 1), ("decode", 1), ("wall", 2))]
            print(f"| {name} | {rows[0]['seed']} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        raise SystemExit("Run without -O so the validation assertions execute.")
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with (ROOT / "runs.csv").open(newline="") as f:
        records = list(csv.DictReader(f))
    assert Counter(r["series"] for r in records) == {"primary": 70, "followup": 24, "startup": 4}
    assert len({r["case"] for r in records}) == len(records)
    rows = [(r, check_run(r)) for r in records]
    primary = [a for r, a in rows if r["series"] == "primary"]
    followup = [a for r, a in rows if r["series"] == "followup"]
    gate = {r["case"].split("/")[-1]: a for r, a in rows if r["series"] == "startup"}
    render(primary, followup, gate, read(ROOT / "profiles.json")["models"], args.per_seed)


if __name__ == "__main__":
    main()
