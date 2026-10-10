# The intermittent slow ruler phase (2026-10-10, TensorFold 1.0.5, the defaults)

Three ruler runs on 1.0.5 had one phase at about half speed with the same tokens and the same draft rounds as a
normal run: structured 61.4 tok/s (`2026-10-10-pool-1.0.5/ruler/ruler.json`), list 41.7
(`ruler/ruler-repeat.json` there) and list 41.8 (`2026-10-10-wiredreads/ruler.json`). Normal: 92 / 105 / 99 / 96.

Looking for it with the server's per-round log (`GLM_WINDOWS=1`):

- `slowhunt.jsonl`: 24 requests (`tools/slowhunt.py`), list and structured prompts, each request's decode rate
  beside its rounds' GPU time, gaps, host and encode time. All 96-107 tok/s. Decode is GPU-bound: GPU time is about
  94% of the wall time, 18-20 ms a round; the largest gap is the idle time before a request's first round.
- `rulers-later.txt`: 8 ruler runs, from 2 minutes after a load. All phases normal.
- `rulers-after-load.txt`: 14 ruler runs starting the moment the server was ready (the three slow runs came 2-6
  minutes after a load). All phases normal.

Not the cause: the time since load; the other local session's GPU tests (its test runs ended 10 minutes before the
first slow run, and it ran nothing around the third); a `powermetrics` GPU sampler (it ran through clean runs too).
Not reproduced in 22 ruler runs and 24 requests; the cause is open.
