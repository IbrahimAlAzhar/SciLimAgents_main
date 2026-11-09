"""
One chat client for every model in the project.

* open models (Qwen3, Mistral-Small-3.1, Gemma-3, Llama) -> local vLLM
  OpenAI-compatible server started by the PBS job (`--llm_base_url`), each
  LoRA adapter addressed by its served name (`model=`).
* GPT models (gpt-4o-mini, ...) -> api.openai.com with OPENAI_API_KEY.
* "mock"  -> deterministic offline generator for dry runs / tests.

Only `requests` is required (no openai SDK). Context is fitted with the
model's own tokenizer so prompt + max_tokens never exceeds the window: the
longer of system/user (the one carrying the paper) is shrunk first.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.text import Tok, strip_think

OPENAI_BASE_URL = "https://api.openai.com/v1"
_tls = threading.local()
_LOG_LOCK = threading.Lock()


def _session():
    import requests
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s


def is_openai_model(name: Optional[str]) -> bool:
    return bool(name) and str(name).startswith(("gpt-", "gpt4", "o1", "o3", "o4", "chatgpt"))


def read_openai_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key and key != "sk-REPLACE_ME":
        return key
    p = os.path.expanduser("~/.openai_key")
    if os.path.exists(p):
        return open(p).read().strip() or None
    return None


@dataclass
class ChatClient:
    model: str                                   # served name / OpenAI model id / "mock"
    base_url: Optional[str] = None               # None + gpt model -> OpenAI
    api_key: Optional[str] = None
    tokenizer: Optional[str] = None              # HF path for exact budgeting
    max_model_len: int = 16384
    family: str = "auto"                         # qwen3 | mistral | gemma | llama | openai | auto
    enable_thinking: bool = False
    timeout: int = 900
    max_retries: int = 4
    merge_system: Optional[bool] = None          # None = auto (retry merged on template error)
    default_repetition_penalty: Optional[float] = None
    _tok: Optional[Tok] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.mock = (self.model == "mock" or self.base_url == "mock")
        self.is_openai = (not self.mock) and (
            (self.base_url is None and is_openai_model(self.model))
            or (self.base_url or "").startswith(OPENAI_BASE_URL))
        if self.is_openai:
            self.base_url = self.base_url or OPENAI_BASE_URL
            self.api_key = self.api_key or read_openai_key()
            if not self.api_key:
                raise RuntimeError(f"{self.model}: OPENAI_API_KEY is not set (export it or put it "
                                   "in ~/.openai_key).")
            self.family = "openai"
            self.max_model_len = max(self.max_model_len, 128000)
        if self.family == "auto":
            low = f"{self.tokenizer or ''} {self.model}".lower()
            self.family = ("qwen3" if "qwen3" in low or "qwen_3" in low else
                           "qwen" if "qwen" in low else
                           "mistral" if "mistral" in low else
                           "gemma" if "gemma" in low else
                           "llama" if "llama" in low else "other")
        self._tok = Tok(None if self.is_openai else self.tokenizer,
                        kind="tiktoken" if self.is_openai else "auto")
        self.base_url = (self.base_url or "").rstrip("/")

    # ------------------------------------------------------------------ utils
    @property
    def tok(self) -> Tok:
        return self._tok

    def count(self, text: str) -> int:
        return self._tok.count(text)

    def truncate(self, text: str, n: int, suffix: str = "... [TRUNCATED]") -> str:
        return self._tok.truncate(text, n, suffix=suffix)

    def fit(self, system: str, user: str, max_tokens: int, overhead: int = 256):
        budget = self.max_model_len - max_tokens - overhead
        ns, nu = self.count(system), self.count(user)
        if ns + nu <= budget:
            return system, user
        # shrink the longer message (it carries the paper) first, then the other
        if ns >= nu:
            system = self.truncate(system, max(512, budget - nu), suffix="\n\n... [TRUNCATED TO FIT CONTEXT]")
            ns = self.count(system)
            if ns + nu > budget:
                user = self.truncate(user, max(256, budget - ns), suffix="\n\n... [TRUNCATED TO FIT CONTEXT]")
        else:
            user = self.truncate(user, max(512, budget - ns), suffix="\n\n... [TRUNCATED TO FIT CONTEXT]")
            nu = self.count(user)
            if ns + nu > budget:
                system = self.truncate(system, max(256, budget - nu), suffix="\n\n... [TRUNCATED TO FIT CONTEXT]")
        return system, user

    @staticmethod
    def merged_messages(system: str, user: str) -> List[Dict[str, str]]:
        if not system:
            return [{"role": "user", "content": user}]
        return [{"role": "user", "content": f"{system}\n\n{user}"}]

    # ------------------------------------------------------------------- core
    def chat(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.2,
             top_p: float = 0.9, seed: Optional[int] = None,
             repetition_penalty: Optional[float] = None, model: Optional[str] = None,
             json_mode: bool = False) -> str:
        model = model or self.model
        system = system or ""
        system, user = self.fit(system, user, max_tokens)
        if self.mock:
            log = os.environ.get("SCILIM_MOCK_LOG")
            if log:
                with _LOG_LOCK, open(log, "a") as f:
                    f.write(json.dumps({"model": model, "system": system[:200], "user": user[:200],
                                        "max_tokens": max_tokens, "temperature": temperature,
                                        "sys_len": len(system), "user_len": len(user)}) + "\n")
            return _mock_generate(system, user, model, seed, json_mode)

        msgs = (self.merged_messages(system, user) if self.merge_system
                else ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": user}])
        payload = {"model": model, "messages": msgs, "max_tokens": int(max_tokens),
                   "temperature": float(temperature), "top_p": float(top_p)}
        if seed is not None:
            payload["seed"] = int(seed)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if not self.is_openai:
            rp = repetition_penalty if repetition_penalty is not None else self.default_repetition_penalty
            if rp is not None:
                payload["repetition_penalty"] = float(rp)
            if self.family == "qwen3":
                payload["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last = None
        for attempt in range(self.max_retries):
            try:
                r = _session().post(f"{self.base_url}/chat/completions", json=payload,
                                    headers=headers, timeout=self.timeout)
                if r.status_code == 400 and "system" in r.text.lower() and self.merge_system is None \
                        and payload["messages"][0]["role"] == "system":
                    # template has no system role (e.g. Mistral-7B v0.x) -> fold it in
                    self.merge_system = True
                    payload["messages"] = self.merged_messages(system, user)
                    continue
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                if r.status_code >= 400:
                    return f"ERROR: HTTP {r.status_code}: {r.text[:500]}"
                out = r.json()["choices"][0]["message"].get("content") or ""
                return out.strip() if self.enable_thinking else strip_think(out)
            except Exception as e:  # noqa: BLE001
                last = e
                sleep = min(30, (2 ** attempt) + random.random())
                print(f"  [retry {attempt+1}/{self.max_retries}] {type(e).__name__}: {str(e)[:200]}"
                      f" — sleeping {sleep:.1f}s", flush=True)
                time.sleep(sleep)
        return f"ERROR: {last}"

    def chat_json(self, system: str, user: str, max_tokens: int = 1024, **kw) -> dict:
        out = self.chat(system, user, max_tokens=max_tokens, temperature=kw.pop("temperature", 0.0),
                        json_mode=self.is_openai or kw.pop("json_mode", False), **kw)
        return parse_json_obj(out)

    def list_models(self) -> List[str]:
        if self.mock:
            return [self.model]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = _session().get(f"{self.base_url}/models", headers=headers, timeout=10)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def wait_ready(self, names=(), max_wait: int = 1800):
        t0 = time.time()
        while time.time() - t0 < max_wait:
            try:
                served = self.list_models()
                missing = [n for n in names if n and n not in served]
                if missing:
                    print(f"[WARN] served models {served} lack {missing} — requests to them will 404.")
                return served
            except Exception:
                time.sleep(10)
        raise RuntimeError(f"LLM server at {self.base_url} never became ready")


def parse_json_obj(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```")).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# Mock generator: deterministic, offline, produces parseable output
# ---------------------------------------------------------------------------
_MOCK_WORDS = ["baseline", "ablation", "dataset", "metric", "seed", "robustness", "novelty",
               "generalization", "reproducibility", "bias", "evaluation", "assumption"]


def _mock_generate(system: str, user: str, model: str, seed, json_mode: bool) -> str:
    h = int(hashlib.md5((system[-300:] + user[-300:] + str(model) + str(seed)).encode()).hexdigest(), 16)
    rnd = random.Random(h)
    blob = (system + " " + user)
    words = [w for w in re.findall(r"[A-Za-z]{6,}", blob)][:400] or _MOCK_WORDS
    if json_mode or "JSON" in user[-600:] or "JSON" in system[-200:]:
        if '"alignment"' in user:
            return json.dumps({"alignment": round(rnd.random(), 3)})
        if '"assignments"' in user:
            n = len(re.findall(r"^\d+\.", user, re.MULTILINE))
            cats = ["novelty_significance", "experimental_evaluation", "data_ethics", "other"]
            return json.dumps({"assignments": [{"index": i + 1, "specialty": rnd.choice(cats)} for i in range(n)]})
        if '"matches"' in user:
            npred = len(re.findall(r"^P\d+\.", user, re.MULTILINE))
            ngt = len(re.findall(r"^G\d+\.", user, re.MULTILINE))
            m = [{"pred": i + 1, "gt": rnd.randint(1, max(ngt, 1)), "confidence": round(rnd.random(), 2)}
                 for i in range(npred) if ngt and rnd.random() < 0.5]
            return json.dumps({"matches": m, "unmatched_pred": [], "unmatched_gt": []})
        if '"judgments"' in user:
            n = len(re.findall(r"^L\d+\.", user, re.MULTILINE))
            return json.dumps({"judgments": [{"index": i + 1, "grounded": rnd.randint(0, 1),
                                              "specific": rnd.randint(0, 1), "valid": rnd.randint(0, 1),
                                              "severity": rnd.randint(0, 2)} for i in range(n)]})
        if '"actionable"' in user:
            return json.dumps({"actionable": rnd.random(), "accurate": rnd.random(), "targeted": rnd.random()})
        if '"organization"' in user:
            return json.dumps({"organization": rnd.random(), "dedup": rnd.random(), "coherence": rnd.random()})
        if '"category"' in user:
            c = rnd.randint(1, 6)
            return json.dumps({"category": c, "valid": "Yes" if c in (1, 2) else "No",
                               "evidence": "mock quote" if c in (1, 2) else "", "reasoning": "mock"})
        return "{}"
    if 'Your answer should be "Yes" or "No"' in user:
        return rnd.choice(["Yes", "No", "No"])
    n = rnd.randint(3, 7)
    lines = ["Here is the consolidated list of key limitations identified in the paper:"]
    cats = ["Novelty & Significance", "Experimental", "Clarity", "Data & Ethics", "Theoretical"]
    for _ in range(n):
        ws = " ".join(rnd.sample(words, k=min(len(words), 10)))
        lines.append(f"- **{rnd.choice(cats)}:** The paper lacks {rnd.choice(_MOCK_WORDS)} analysis regarding {ws}.")
    return "\n".join(lines)
