"""Replay one or more of the recorded Gufo requests on a running server."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen


def read_stream(wire, request):
    frames = [line[6:] for line in wire.splitlines() if line.startswith(b"data: ")]
    if not frames or frames.pop() != b"[DONE]":
        raise ValueError("incomplete SSE response")
    legacy = "prompt" in request
    content, reasoning, covered = bytearray(), bytearray(), bytearray()
    scores = 0
    usage = timings = finish = None
    for line in frames:
        frame = json.loads(line)
        usage = frame.get("usage") or usage
        timings = frame.get("timings") or timings
        for choice in frame["choices"]:
            delta = choice.get("delta") or {}
            visible = choice.get("text") if legacy else delta.get("content")
            content.extend((visible or "").encode())
            reasoning.extend((delta.get("reasoning_content") or "").encode())
            finish = choice.get("finish_reason") or finish
            probabilities = choice.get("logprobs") or {}
            if legacy:
                values = probabilities.get("token_logprobs") or []
                assert len(values) == len(probabilities.get("tokens") or [])
                assert all(math.isfinite(v) and v <= 0 for v in values)
                scores += len(values)
            else:
                for entry in probabilities.get("content") or []:
                    ranks = [token["logprob"] for token in entry["top_logprobs"]]
                    assert math.isfinite(entry["logprob"]) and entry["logprob"] <= 0
                    assert all(math.isfinite(rank) and rank <= 0 for rank in ranks)
                    assert ranks == sorted(ranks, reverse=True) and len(ranks) == request.get("top_logprobs", 0)
                    covered.extend(entry["bytes"])
                    scores += 1
    if not usage or not finish:
        raise ValueError("stream missing final usage or finish reason")
    if not legacy and request.get("logprobs"):
        assert covered == content
    if "logprobs" not in request:
        assert scores == 0
    return dict(usage=usage, timings=timings, finish=finish, scored_tokens=scores,
                content_hash=hashlib.sha256(content).hexdigest(), content_bytes=len(content),
                reasoning_hash=hashlib.sha256(reasoning).hexdigest(), reasoning_bytes=len(reasoning),
                wire_bytes=len(wire), sse_events=len(frames))


def phase_values(stream):
    usage = stream["usage"]
    if "gufo" in usage:
        phases = usage["gufo"]
        return phases["prefill_tokens"], phases["prefill_ms"], usage["completion_tokens"], phases["decode_ms"]
    phases = stream["timings"]
    return phases["prompt_n"], phases["prompt_ms"], phases["predicted_n"], phases["predicted_ms"]


def replay(base_url, request, served_model):
    if served_model:
        request["model"] = served_model
    path = "/completions" if "prompt" in request else "/chat/completions"
    http_request = Request(base_url.rstrip("/") + path, data=json.dumps(request, ensure_ascii=False).encode(),
                           headers={"Content-Type": "application/json"})
    lines = []
    started = time.perf_counter()
    with urlopen(http_request, timeout=900) as response:
        if response.headers.get_content_type() != "text/event-stream":
            raise ValueError("expected an SSE response")
        for line in response:
            lines.append(line)
            if line.strip() == b"data: [DONE]":
                break
        wall = time.perf_counter() - started
    stream = read_stream(b"".join(lines), request)
    prefill_tokens, prefill_ms, generated, decode_ms = phase_values(stream)
    return dict(model=request["model"], seed=request["seed"], prefill_tok_s=prefill_tokens * 1000 / prefill_ms,
                decode_tok_s=generated * 1000 / decode_ms, wall_s=wall, output_tokens=generated,
                cached_tokens=stream["usage"]["prompt_tokens_details"]["cached_tokens"],
                scored_tokens=stream["scored_tokens"], finish=stream["finish"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8900/v1", help="OpenAI-compatible API base including /v1")
    parser.add_argument("--model", help="override the served model ID in the recorded requests")
    parser.add_argument("requests", nargs="+", type=Path, help="one or more streaming request.json files, in execution order")
    args = parser.parse_args()
    for path in args.requests:
        print(json.dumps(dict(request=str(path), **replay(args.base_url, json.loads(path.read_text()), args.model))), flush=True)
