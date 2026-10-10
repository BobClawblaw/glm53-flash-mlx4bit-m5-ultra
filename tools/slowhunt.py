"""Each request's decode rate beside its rounds from the server's GLM_WINDOWS log: where a slow phase's time goes.

Serve with `extra_env = "GLM_WINDOWS=1"` (local.toml [engine]), then: python3 -I tools/slowhunt.py [repeats]
"""
import json, os, re, statistics, sys, time, urllib.request
LOG = os.path.expanduser("~/.cache/tensorfold-glm53/server.log")  # the recipe's server log; serve with GLM_WINDOWS=1
U = "http://127.0.0.1:8888/v1/chat/completions"
PROMPTS = {
    "list": "List 40 {x}, numbered, one per line, no commentary.",
    "structured": "Return a JSON object describing {x}: fields name, category, five key facts (array of strings), and a numeric rating from 1 to 10. JSON only.",
}
ITEMS = {"list": ["rivers", "chemical elements", "programming languages", "capital cities", "birds", "card games"],
         "structured": ["the Nile", "the ukulele", "Mount Fuji", "the octopus", "Saturn", "chess"]}
W = re.compile(r"glm window: streams (\d+) rows (\d+) gpu ([\d.]+) ms gap ([\d.]+) ms ahead ([\d.]+) join ([\d.]+) host ([\d.]+) encode ([\d.]+)")
def run(prompt):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = os.path.getsize(LOG); t0 = time.time(); first = None; last = None; n = 0
    for line in urllib.request.urlopen(req, timeout=300):
        if line.startswith(b"data: {"):
            d = json.loads(line[6:])
            if d.get("usage"): n = d["usage"]["completion_tokens"]
            if d.get("choices") and d["choices"][0]["delta"].get("content"):
                last = time.time(); first = first or last
    time.sleep(0.3)
    with open(LOG, "rb") as f:
        f.seek(start); text = f.read().decode(errors="replace")
    w = [tuple(float(x) for x in m.groups()) for m in W.finditer(text)]
    rate = (n - 1) / (last - first) if first and last > first else 0
    return {"tok_s": rate, "tokens": n, "rounds": len(w), "rows": sum(x[1] for x in w), "gpu_ms": sum(x[2] for x in w),
            "gap_ms": sum(x[3] for x in w), "max_gap": max((x[3] for x in w), default=0), "host_ms": sum(x[6] for x in w),
            "encode_ms": sum(x[7] for x in w), "wall_ms": (last - first) * 1e3 if first else 0,
            "gpu_per_round": statistics.mean(x[2] for x in w) if w else 0}
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for r in range(reps):
    for phase, tmpl in PROMPTS.items():
        for x in ITEMS[phase][:3]:
            m = run(tmpl.format(x=x))
            print(json.dumps({"rep": r, "phase": phase, "item": x, **{k: round(v, 2) for k, v in m.items()}}), flush=True)
