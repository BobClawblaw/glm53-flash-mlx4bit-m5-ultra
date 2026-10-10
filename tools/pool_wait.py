"""The shared pool under pressure: A reserves nearly the whole pool, B must wait for it; C, larger than the window, is refused."""
import json, threading, time, urllib.request, urllib.error
U = "http://127.0.0.1:8888/v1/chat/completions"
T0 = time.time()
def chat(name, content, max_tokens, out):
    body = {"model": "x", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time() - T0
    try:
        d = json.load(urllib.request.urlopen(req, timeout=600))
        out[name] = (round(t, 2), round(time.time() - T0, 2), d["choices"][0]["message"]["content"][:60], d["usage"])
    except urllib.error.HTTPError as e:
        out[name] = (round(t, 2), round(time.time() - T0, 2), e.code, e.read()[:160])
out = {}
# A reserves ~1,040,000 of the 1,048,572-token pool; it writes ~300 tokens then stops
a = threading.Thread(target=chat, args=("A", "Write the numbers from 1 to 120 separated by spaces, nothing else.", 1048000, out))
a.start(); time.sleep(0.4)
b = threading.Thread(target=chat, args=("B", "Reply with exactly: pool ok", 2000, out)); b.start()
time.sleep(0.3)
s = json.load(urllib.request.urlopen("http://127.0.0.1:8888/health"))["live"]
print("during A:", s)
a.join(); b.join()
chat("C", "hi", 1048570, out)  # more than the pool: refused
for k in sorted(out): print(k, out[k])
