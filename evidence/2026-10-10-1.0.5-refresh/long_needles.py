"""Needles at nominal 600k and 1.15M tokens on 1.0.5 (the suite's needle(), thinking off, depth 0.5), each length on
its own seed so no document's opening is another's and the prompt cache resumes nothing.

python3 -I evidence/2026-10-10-1.0.5-refresh/long_needles.py .external/spark-recipe http://127.0.0.1:8888
"""
import json, sys
sys.path.insert(0, sys.argv[1])
from quality.common import Client  # noqa: E402
from quality.needles import needle  # noqa: E402
client = Client(sys.argv[2], "x")
rows = []
for nominal, seed in ((600_000, 23), (1_150_000, 29)):
    r = needle(client, nominal, 0.5, seed=seed)
    rows.append({"nominal": nominal, "seed": seed, **r})
    print(f"nominal {nominal}: {r['tokens']} prompt tokens, found={r['found']}, {r['seconds']:.0f} s, prefill {r['prefill_tok_s']:.0f} tok/s", flush=True)
json.dump(rows, open(sys.argv[0].replace("long_needles.py", "needles.json"), "w"), indent=1)
