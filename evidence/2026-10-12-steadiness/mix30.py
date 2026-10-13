"""30 single requests: code, prose and edit prompts, some repeated (prompt-cache hits), greedy, 256 tokens."""
import json, urllib.request
U = "http://127.0.0.1:8888/v1/chat/completions"
P = ["Write a Python function for Fibonacci numbers with memoization, then explain it.",
     "Write a short story about a lighthouse keeper.",
     "Rewrite this sentence to be more formal: hey can u send me the report by friday thx",
     "Explain how a hash table handles collisions.",
     "Write a haiku sequence about autumn rain."]
out = []
for i in range(30):
    b = {"model": "x", "messages": [{"role": "user", "content": P[i % 5] + ("" if i < 15 else " Be concise.")}], "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False}}
    o = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600))
    out.append((o.get("runtime") or o.get("tensorfold") or {}).get("tokens_per_second"))
print(" ".join(f"{x:.0f}" for x in out))
same = lambda i: [out[j] for j in range(30) if j % 5 == i % 5 and (j < 15) == (i < 15)]
slow = [i for i, x in enumerate(out) if x < 0.85 * max(same(i))]
print(f"replies under 85% of the same prompt's best: {len(slow)} of 30 {slow}")
