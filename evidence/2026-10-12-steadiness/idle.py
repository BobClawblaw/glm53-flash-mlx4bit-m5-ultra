"""First token after the server sat idle 10 s and 30 s, streamed, three times each."""
import json, time, urllib.request
U = "http://127.0.0.1:8888/v1/chat/completions"
def ttft():
    b = {"model": "x", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 16, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=120) as r:
        for line in r:
            if b'"content"' in line:
                return time.time() - t
    return time.time() - t
for idle in (10, 30):
    got = []
    for _ in range(3):
        time.sleep(idle)
        got.append(ttft())
    print(f"after {idle} s idle: first token " + ", ".join(f"{x:.2f} s" for x in got), flush=True)
