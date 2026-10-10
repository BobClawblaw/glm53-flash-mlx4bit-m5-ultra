"""Long prompts (the sparse attention path) in pool slices at offsets, beside a 24k-token stream, against each alone.

python3 -I tools/pool_offsets.py .external/spark-recipe   (the suite clone, for its needle documents)
"""
import json, sys, threading, time, urllib.request
sys.path.insert(0, sys.argv[1])
from quality.needles import document
U = "http://127.0.0.1:8888/v1/chat/completions"
def prompt(tokens, seed, code):
    p = document(tokens, seed); p.insert(len(p) * 2 // 3, f"The secret code for the blue door is {code}. Remember it.\n\n")
    return "Read this document carefully.\n\n" + "".join(p) + "\n\nQuestion: What is the secret code for the blue door, and what do sections 3, 4 and 5 report? Answer in three short sentences."
def chat(content, mt, out, k):
    body = {"model": "x", "messages": [{"role": "user", "content": content}], "max_tokens": mt, "chat_template_kwargs": {"enable_thinking": False}}
    d = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
    out[k] = (d["usage"]["prompt_tokens"], d["choices"][0]["message"]["content"])
X = prompt(12000, 3, "4-1-7-7-2"); Y = prompt(30000, 5, "8-8-3-0-6"); Z = prompt(9000, 9, "5-2-6-1-3")
alone = {}
for k, p in (("X", X), ("Z", Z)): chat(p, 120, alone, k)
tog = {}
ts = [threading.Thread(target=chat, args=(Y, 120, tog, "Y"))]; ts[0].start(); time.sleep(0.5)
for k, p in (("X", X), ("Z", Z)):
    t = threading.Thread(target=chat, args=(p, 120, tog, k)); t.start(); ts.append(t); time.sleep(0.3)
for t in ts: t.join()
for k in ("X", "Z"):
    print(k, alone[k][0], "tokens; identical together vs alone:", alone[k][1] == tog[k][1]); print("   ", alone[k][1][:150].replace("\n", " "))
print("Y", tog["Y"][0], tog["Y"][1][:100].replace("\n", " "))
