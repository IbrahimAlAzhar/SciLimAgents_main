"""
LimAgents + NovAgents merge prompts.

anchor     : Report A (LimAgents) is the anchor, 10-14 limitations
             (merge_and_eval.py version — items 1 and 2 correctly separated).
anchor_v0  : byte-identical copy from limagents+novagents/merge.py where a
             missing "." + newline fused items 1 and 2. Use only to reproduce
             old numbers exactly.
simple     : pairwise merge without count constraint (lim_nov.py).
"""
from ._raw import (MERGE_ANCHOR_SYSTEM_PROMPT, MERGE_ANCHOR_SYSTEM_PROMPT_V0,
                   MERGE_SIMPLE_SYSTEM_PROMPT, MERGE_SIMPLE_USER_TEMPLATE)


def get_merge_user_prompt(lim_text: str, nov_text: str) -> str:
    return f"""\
### Report A — Limitation Agents (primary, must be kept):
{lim_text}

### Report B — Novelty Agents (merge into A when similar, otherwise keep separate):
{nov_text}

Produce the consolidated limitation report following the instructions provided.
"""


MERGE_STYLES = {
    "anchor": (MERGE_ANCHOR_SYSTEM_PROMPT, get_merge_user_prompt),
    "anchor_v0": (MERGE_ANCHOR_SYSTEM_PROMPT_V0, get_merge_user_prompt),
    "simple": (MERGE_SIMPLE_SYSTEM_PROMPT,
               lambda lim, nov: MERGE_SIMPLE_USER_TEMPLATE.format(lim_text=lim, nov_text=nov)),
}


def build_merge_messages(lim_text: str, nov_text: str, style: str = "anchor"):
    system, user_fn = MERGE_STYLES[style]
    return system, user_fn(lim_text, nov_text)
