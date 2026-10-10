"""Kept prompt states in the shared pool: reused after their stream leaves, from another slot, and given up under pressure.

python3 -I tools/pool_reuse.py .external/spark-recipe   (the suite clone, for its generated documents)

1. A long system prompt and a first question: nothing to reuse.
2. A second question on the same prompt while another stream holds slot 0: the kept state's prefix comes from the
   pool blocks its finished stream left held, into another slot.
3. A request that takes nearly the whole pool: the held blocks give way (it must not wait for them).
4. The second question again: the state went stale, the prompt is read again, and the reply equals step 2's.
"""

import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, sys.argv[1])
from quality.needles import document  # noqa: E402

U = "http://127.0.0.1:8888/v1/chat/completions"
SYSTEM = "You answer questions about this document.\n\n" + "".join(document(8000, 5))


def chat(messages, max_tokens):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    u = d["usage"]
    return d["choices"][0]["message"]["content"], u["prompt_tokens"], u.get("prompt_tokens_details", {}).get("cached_tokens", 0), time.time() - t


def ask(q):
    return chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}], 80)


q1 = "What does section 2 report? One sentence."
q2 = "What does section 7 report? One sentence."
r1 = ask(q1)
print(f"1 first question: {r1[1]} prompt tokens, {r1[2]} cached, {r1[3]:.2f} s")
busy = {}
t = threading.Thread(target=lambda: busy.update(r=chat([{"role": "user", "content": "Count from 1 to 400, spaces only."}], 1600)))
t.start()
time.sleep(1.0)  # the counting stream holds a slot before the second question arrives
r2 = ask(q2)
t.join()
print(f"2 second question beside another stream: {r2[2]} of {r2[1]} cached, {r2[3]:.2f} s")
r3 = chat([{"role": "user", "content": "Reply with exactly: big ok"}], 1048000)
print(f"3 nearly the whole pool: {r3[0]!r} in {r3[3]:.2f} s")
r4 = ask(q2)
print(f"4 second question again: {r4[2]} of {r4[1]} cached, {r4[3]:.2f} s")
print(f"reply 4 equals reply 2: {r4[0] == r2[0]}")
print(f"reply 2: {r2[0][:120]!r}")
