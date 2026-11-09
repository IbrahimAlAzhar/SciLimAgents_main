"""Text helpers shared by every pipeline (cleaning, citation extraction,
bullet parsing, token budgeting)."""
from __future__ import annotations

import ast
import re
import threading
from typing import List, Optional

import pandas as pd

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    if not text:
        return ""
    text = THINK_RE.sub(" ", str(text))
    # unterminated <think> (generation cut off inside the reasoning block)
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "pending"}


def as_text(x) -> str:
    return "" if is_missing(x) else str(x)


def clean_text_detailed(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("\n", " ")
    text = re.sub(r"\S+\s+et\s+al\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_cited(cited_entry):
    if cited_entry is None or (isinstance(cited_entry, float) and pd.isna(cited_entry)):
        return None
    try:
        parsed = ast.literal_eval(cited_entry) if isinstance(cited_entry, str) else cited_entry
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_intro_and_abstract(cited_entry, style: str = "v1", top_k: Optional[int] = None) -> str:
    """Cited-paper context from the `cited_in` / `ret_abs_int_cit` cell.

    style="v1"      : limagents.py / dpo inference format, all papers,
                      "'Paper1_Title: ..', 'Paper1_Abstract': '..', ..." one line each
    style="rollout" : rollout_qwen_vllm.py format, top_k papers, 3 lines each
    """
    parsed = _parse_cited(cited_entry)
    if not parsed:
        return ""
    processed = []
    for idx, (_, data) in enumerate(parsed.items(), 1):
        if top_k is not None and idx > top_k:
            break
        if not isinstance(data, dict):
            continue
        intro = ""
        for sec in data.get("sections", []) or []:
            if not isinstance(sec, dict):
                continue
            if "introduction" in str(sec.get("heading", "")).lower():
                intro = sec.get("text", "")
                break
        t_clean = clean_text_detailed(data.get("title", ""))
        a_clean = clean_text_detailed(data.get("abstractText") or data.get("abstract"))
        i_clean = clean_text_detailed(intro)
        if t_clean or a_clean or i_clean:
            if style == "rollout":
                processed.append(f"Paper{idx}_Title: {t_clean}\nPaper{idx}_Abstract: {a_clean}\n"
                                 f"Paper{idx}_Introduction: {i_clean}")
            else:
                processed.append(
                    f"'Paper{idx}_Title: {t_clean}', "
                    f"'Paper{idx}_Abstract': '{a_clean}', "
                    f"'Paper{idx}_Introduction': '{i_clean}'."
                )
    return ("\n\n" if style == "rollout" else "\n").join(processed)


def parse_relevant_papers_list(x) -> List[str]:
    """limagents.py version: cell -> list[str] (drops empty items)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i) for i in parsed if str(i).strip()]
            return [str(parsed)]
        except Exception:
            return [s]
    return [str(x)]


def parse_relevant_list_nov(x):
    """novagents.py version (keeps items as-is)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x.strip())
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [x.strip()]
    return [str(x)]


def build_evidence_tagged(paper_text: str) -> str:
    """Tag sentences with [E#] IDs so agents can cite evidence IDs in strict mode."""
    chunks = re.split(r"(?<=[.!?])\s+", paper_text)
    tagged, n = [], 0
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        n += 1
        tagged.append(f"[E{n}] {c}")
    return " ".join(tagged)


_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]\s|\Z)", re.DOTALL | re.MULTILINE)


def split_bullets(text: str) -> List[str]:
    if not text:
        return []
    text = strip_think(text)
    items = [re.sub(r"\s+", " ", m).strip() for m in _BULLET_RE.findall(text)]
    if not items:
        items = re.split(r"\n\s*\d+\.\s+|\n\n+", text)
        items = [re.sub(r"\s+", " ", i).strip() for i in items if i.strip()]
    return [i for i in items if len(i) > 20]


# ---------------------------------------------------------------------------
# Tokenizer wrapper (HF tokenizer, tiktoken, or char fallback)
# ---------------------------------------------------------------------------
class Tok:
    """Uniform encode/decode/count over an HF tokenizer, tiktoken, or chars/4."""

    def __init__(self, path_or_name: Optional[str] = None, kind: str = "auto"):
        self.kind = "chars"
        self._tok = None
        self._lock = threading.Lock()
        self.path = path_or_name
        if kind in ("auto", "hf") and path_or_name and not str(path_or_name).startswith(("gpt", "o1", "o3", "o4")):
            try:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
                self.kind = "hf"
            except Exception as e:  # noqa: BLE001
                if kind == "hf":
                    raise
                print(f"[Tok] HF tokenizer unavailable for {path_or_name} ({e}); trying tiktoken")
        if self._tok is None and kind in ("auto", "tiktoken"):
            try:
                import tiktoken
                try:
                    self._tok = tiktoken.get_encoding("o200k_base")
                except Exception:
                    self._tok = tiktoken.get_encoding("cl100k_base")
                self.kind = "tiktoken"
            except Exception:
                self._tok = None
                self.kind = "chars"

    def encode(self, text: str) -> list:
        text = str(text or "")
        if self.kind == "hf":
            return self._tok.encode(text, add_special_tokens=False)
        if self.kind == "tiktoken":
            return self._tok.encode(text, disallowed_special=())
        return list(range((len(text) + 3) // 4))

    def decode(self, ids) -> str:
        if self.kind == "hf":
            return self._tok.decode(ids, skip_special_tokens=True)
        if self.kind == "tiktoken":
            return self._tok.decode(ids)
        raise RuntimeError("decode not supported in chars mode")

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, max_tokens: int, suffix: str = "... [TRUNCATED]",
                 keep: str = "head") -> str:
        """Hard-truncate to max_tokens tokens (suffix appended when cut)."""
        text = str(text or "")
        if not text:
            return ""
        if self.kind == "chars":
            lim = max_tokens * 4
            if len(text) <= lim:
                return text
            return (text[-lim:] if keep == "tail" else text[:lim]) + suffix
        ids = self.encode(text)
        if len(ids) <= max_tokens:
            return text
        ids = ids[-max_tokens:] if keep == "tail" else ids[:max_tokens]
        return self.decode(ids) + suffix

    @property
    def hf(self):
        return self._tok if self.kind == "hf" else None
