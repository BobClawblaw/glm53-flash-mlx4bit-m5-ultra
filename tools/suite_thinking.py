"""The Spark recipe's quality suite with thinking on: python3 tools/suite_thinking.py BUDGET SUITE_ARGS... (from its clone).

The suite sends every chat request with thinking off and reply limits sized for answers alone. This wrapper rewrites
each request to /v1/chat/completions at the HTTP layer, so all of the suite's paths (its client and the streaming
check's own requests) are covered and the pinned suite stays untouched:
  - chat_template_kwargs.enable_thinking = true;
  - thinking_budget = BUDGET (the engine closes the think block after that many tokens);
  - max_tokens += BUDGET (the reasoning counts against the reply limit; the answer keeps its own room).
The behaviour check's exact-max_tokens case measures the raised limit, so read it with that in mind.
"""

from __future__ import annotations

import json
import sys
import urllib.request

BUDGET = int(sys.argv[1])
_urlopen = urllib.request.urlopen


def _thinking_urlopen(req, *args, **kwargs):
    if isinstance(req, urllib.request.Request) and req.full_url.endswith("/v1/chat/completions") and req.data:
        body = json.loads(req.data)
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
        body["thinking_budget"] = BUDGET
        body["max_tokens"] = int(body.get("max_tokens") or 512) + BUDGET
        req.data = json.dumps(body).encode()
        req.headers.pop("Content-length", None)
    return _urlopen(req, *args, **kwargs)


urllib.request.urlopen = _thinking_urlopen
sys.path.insert(0, ".")
from quality import suite  # noqa: E402

sys.exit(suite.main(sys.argv[2:]))
