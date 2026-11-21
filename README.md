# Checking the Gufo logprobs timings on Strix Halo

These are the 98 requests behind the PR tables. You can check the numbers with Python 3 and its standard library. From the unpacked `gufo-logprobs-bench` directory, run:

```sh
python3 verify.py --per-seed
```

The script checks the SHA-256 manifests, reads every captured HTTP stream, and recalculates prefill and decode tokens/s from token counts and milliseconds. It checks score coverage, text hashes, cache hits, finish reasons and speculative token counts. Wall time is the original client's monotonic measurement through the SSE `[DONE]` marker. The stream records the server's phases. `runs.csv` records the independent client clock.

## Repeat a request

`profiles.json` contains the exact Gufo arguments, NPU settings, binary digests and GGUF repositories, revisions and hashes for each profile. Map `/weights` to your own weights directory. Both builds ran on the same machine with the same quantizations and the two small GLM and Laguna fixes under `build/`. `build/startup-flag.patch` applies to the review commit listed in `profiles.json`.

Start the matching server with one session and a 16,384-token context. On the gated build add `--allow-logprobs` for the scored runs. Warm the process once, then replay the recorded requests. For example, with the Qwen 35B profile running on port 8900:

```sh
mid=gufo-qwen3.6-35b-a3b-npu
curl -fsS -H 'Content-Type: application/json' --data-binary @warmup/$mid.json \
  http://127.0.0.1:8900/v1/chat/completions >/dev/null
python3 replay.py --base-url http://127.0.0.1:8900/v1 \
  runs/primary/$mid/seed-43-new-off/request.json \
  runs/primary/$mid/seed-43-new-k10/request.json
```

Each request prints prefill, decode, wall time, output length, cached tokens and score count. `replay.py --model NAME` can override the served ID if you also change the model ID in the warm-up JSON. The two-request example checks a single seed. To repeat the full comparison, follow `runs.csv` in timestamp order, restarting and warming the process whenever its model or server segment changes, also at the start of each series. For the four startup runs, restart for each row and enable the startup flag only for `candidate-permitted`. The pre-flag review binary already accepted per-request score selectors. Keep downloads, model loads and warm-ups outside the clock.

## What the three series cover

- **Primary, 70 requests.** Seven models, three seeds (42, 43, 44), three conditions and one repeat of the unscored request per model. `old-off` is the fork base, `new-off` is the reviewed backport with no requested scores and `new-k10` uses the same backport with ten alternatives. Each condition ran first, second and third once per model. We used the cache-disabled, vendor-sampling settings recorded in each request. Gufo processed 11,038–11,444 input tokens and generated 1,024 tokens per request.
- **Follow-up, 24 requests.** We ran off, chosen-token only and top ten in one warm process at seed 43 for each model. Flint also got the same three conditions through raw Completions because its Chat template used all 1,024 tokens for reasoning. The raw scored requests exposed all 1,024 token scores.
- **Startup flag, four requests.** Qwen 35B ran the same ordinary request on the pre-flag review build, the new default, the new build with the flag enabled, then the review build again. Each process started warm. All four runs produced the same visible text, reasoning and draft counts.

Each table reports median rates across the three primary seeds. Percentage changes are medians of paired within-seed changes. The per-request slowdown test flags a phase when its duration increases by more than `max(3 ms, 5% of the reference duration)`. A longer run or a different prompt can give a different result. In particular, Flint also varied when we repeated the identical seed on the same binary.
