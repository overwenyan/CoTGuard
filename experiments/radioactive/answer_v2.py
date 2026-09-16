"""Corrected GSM8K answer extraction (2026-09-16 integrity fix).

utility_check.extract_answer (v1) has a fallback lookahead (?![\\d.]) that skips any number followed by a
period, so "The answer is 8." falls back to an earlier number (e.g. the 3 in "Step 3"). v1 is left unchanged
because every historical number was produced with it; v2 is used to quantify the bias and for new checks.
"""

from __future__ import annotations

import re

NUM = r"-?\$?\s*[\d,]*\d(?:\.\d+)?"
PATTERNS = [
    re.compile(r"####\s*\$?\s*(-?[\d,]*\d(?:\.\d+)?)"),
    re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\d(?:\.\d+)?)\s*\}"),
    re.compile(r"(?:the\s+(?:final\s+)?answer\s+is|final\s+answer\s*[:：]?)[^\d\-\n]{0,20}(-?[\d,]*\d(?:\.\d+)?)", re.I),
]
FALLBACK = re.compile(r"(?<![\d.])(-?[\d,]*\d(?:\.\d+)?)(?!\d|\.\d)")


def extract_answer_v2(text: str):
    for pat in PATTERNS:
        m = pat.findall(text)
        if m:
            try:
                return float(m[-1].replace(",", ""))
            except ValueError:
                continue
    m = FALLBACK.findall(text)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def correct_v2(text, gold):
    from utility_check import gold_answer
    x, g = extract_answer_v2(text), gold_answer(gold)
    return x is not None and g is not None and abs(x - g) < 1e-6


if __name__ == "__main__":
    for t, want in [("Step 3: State the answer.\nThe answer is 8.", 8), ("Money left = $30 - $21 = $9.", 9),
                    ("#### 1,234", 1234), ("so \\boxed{42}", 42), ("Final Answer: 17 apples", 17),
                    ("He has 3.5 kg.", 3.5), ("Total is 12.\n", 12)]:
        got = extract_answer_v2(t)
        assert got == want, (t, got)
    print("answer_v2 self-test ok")
