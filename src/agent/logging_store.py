"""Append-only JSONL log of every agent turn: question, tool call(s), raw
tool JSON, template used, final response, and the groundedness check
result. Required by principle 4 ("every response must be evaluable after
the fact") -- this is what makes that possible, and what the eval suite
(src/agent/eval/) reads back when reviewing runtime traffic rather than its
own fresh eval-set runs.
"""

import json
import os

LOG_DIR = "data/generated/agent_logs"
LOG_PATH = os.path.join(LOG_DIR, "conversations.jsonl")


def append_log(log: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log, default=str) + "\n")
