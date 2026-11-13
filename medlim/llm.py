"""
Async chat client for an OpenAI-compatible endpoint: a local vLLM server
(open models) or the OpenAI API (gpt-4o-mini). backend='mock' returns canned
output for offline plumbing tests only.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time


class FatalAPIError(RuntimeError):
    pass


class ChatLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 backend: str = "openai", max_concurrency: int = 16, timeout: float = 600,
                 disable_thinking: bool = False, json_mode: bool = False, seed: int = 0):
        self.model, self.backend, self.json_mode, self.seed = model, backend, json_mode, seed
        self.sem = asyncio.Semaphore(max_concurrency)
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None
        self.n_calls = self.n_errors = self.tok_in = self.tok_out = 0
        self.client = None
        if backend == "openai":
            from openai import AsyncOpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if base_url else None)
            if not key:
                raise SystemExit("OPENAI_API_KEY is not set")
            self.client = AsyncOpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0)

    async def chat(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.0,
                   tries: int = 6, json_mode: bool | None = None) -> str:
        """Returns the reply text ('' after repeated failure)."""
        json_mode = self.json_mode if json_mode is None else json_mode
        async with self.sem:
            if self.backend == "mock":
                self.n_calls += 1
                return mock_reply(system, user)
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
            kw = dict(model=self.model, messages=msgs, max_tokens=max_tokens, temperature=temperature)
            if self.seed is not None:
                kw["seed"] = self.seed
            if self.extra_body:
                kw["extra_body"] = self.extra_body
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            last = ""
            for t in range(tries):
                try:
                    r = await self.client.chat.completions.create(**kw)
                    self.n_calls += 1
                    if getattr(r, "usage", None):
                        self.tok_in += r.usage.prompt_tokens or 0
                        self.tok_out += r.usage.completion_tokens or 0
                    return (r.choices[0].message.content or "").strip()
                except Exception as e:  # noqa: BLE001
                    code = getattr(e, "status_code", None)
                    last = f"{type(e).__name__}: {str(e)[:300]}"
                    if code in (401, 403) or "invalid_api_key" in str(e) or "insufficient_quota" in str(e):
                        raise FatalAPIError(last)
                    if code == 400:
                        if "response_format" in kw:          # server without guided JSON -> retry without
                            kw.pop("response_format")
                            continue
                        if "maximum context length" in str(e) or "max_tokens" in str(e):
                            # prompt too long: shrink the user message by 30% and retry
                            u = msgs[-1]["content"]
                            msgs[-1] = {"role": "user", "content": u[: int(len(u) * 0.7)]}
                            kw["messages"] = msgs
                            continue
                    await asyncio.sleep(min(60, 2 ** t) + random.random())
            self.n_errors += 1
            print(time.strftime("[%H:%M:%S]"), f"[llm:{self.model}] gave up after {tries} tries: {last}", flush=True)
            return ""

    def stats(self) -> dict:
        return {"model": self.model, "calls": self.n_calls, "errors": self.n_errors,
                "prompt_tokens": self.tok_in, "completion_tokens": self.tok_out}


# ----------------------------------------------------------------------------- offline mock
_MOCK_POOL = [
    "The sample was drawn from a single academic center, limiting generalizability to rural populations.",
    "Black and Hispanic patients were underrepresented relative to national disease prevalence.",
    "No subgroup performance analysis by age, sex or race was reported.",
    "The retrospective design cannot rule out residual confounding.",
    "Follow-up duration was too short to assess long-term outcomes.",
    "The intervention assumes specialist availability that is lacking in HRSA shortage areas.",
    "Self-reported outcomes are subject to recall bias.",
    "The validation cohort overlaps in time with the training cohort, risking temporal leakage.",
]


def mock_reply(system: str, user: str) -> str:
    h = int(hashlib.md5((system + user[:2000]).encode()).hexdigest(), 16)
    rnd = random.Random(h)
    if "Score each item" in user:
        g = re.findall(r"\[(G\d+)\]", user)
        return json.dumps({"scores": [{"g": x, "score": rnd.randint(0, 2), "data_grounded": rnd.random() < .3} for x in g]})
    if "VERBATIM quote" in user and "acknowledge" in user:
        body = user.split("PAPER TEXT", 1)[-1]
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if 6 <= len(s.split()) <= 40][:60]
        pick = rnd.sample(sents, min(len(sents), rnd.randint(1, 4))) if sents else []
        return json.dumps({"limitations": [{"limitation": "Authors note: " + q, "quote": q} for q in pick]})
    if "LEADER" in system.upper():
        c = re.findall(r"\[(C\d+)\]", user)
        return json.dumps({"limitations": [{"limitation": p, "societal_impact": "Rural patients may be misserved.",
                                            "affected_population": "rural adults", "source_ids": c[:2]}
                                           for p in rnd.sample(_MOCK_POOL, 3)]})
    if "CLUSTER" in system.upper() and "MAJORITY" in system.upper():
        ids = re.findall(r"\[([A-Z]\d+_\d+)\]", user)
        groups = {}
        for i in ids:
            groups.setdefault(int(i.split("_")[1]) % 4, []).append(i)
        return json.dumps({"clusters": [{"canonical": _MOCK_POOL[k], "members": v, "duplicates_author": k == 0}
                                        for k, v in groups.items()]})
    if "Respond with JSON only" in user and "matches" in user:
        g = re.findall(r"\[(G\d+)\]", user)
        r = re.findall(r"\[(R\d+)\]", user)
        return json.dumps({"matches": [{"g": x, "r": rnd.sample(r, k=min(len(r), rnd.randint(0, 1)))} for x in g]})
    if 'Your answer should be "Yes" or "No"' in user:
        return rnd.choice(["Yes", "No", "No"])
    k = rnd.randint(2, 5)
    pick = rnd.sample(_MOCK_POOL, k)
    style = rnd.random()
    if "MERGE" in system.upper():
        return json.dumps({"limitations": [{"claim": p, "source_ids": ["C1"]} for p in pick[:3]]})
    if style < 0.6:
        if "evidence_span" in user:
            span = user[user.find("PAPER TEXT"):][40:140] if "PAPER TEXT" in user else "n/a"
            return json.dumps({"limitations": [{"dimension": "cohort", "claim": p, "evidence_span": span,
                                                "evidence_source": "manuscript", "severity": "high",
                                                "confidence": 0.8} for p in pick]})
        return "```json\n" + json.dumps({"limitations": [{"limitation": p} for p in pick]}) + "\n```"
    if style < 0.85:
        return "Here are the limitations:\n" + "\n".join(f"Limitation {i + 1}: {p}" for i, p in enumerate(pick))
    return "\n".join(f"- {p}" for p in pick)
