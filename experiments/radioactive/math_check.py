"""MATH answer checking for M7: last \\boxed{} of output vs last \\boxed{} of the reference solution."""

from __future__ import annotations

import re


def last_boxed(text: str):
    i = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if i < 0:
        return None
    j = text.index("{", i) + 1
    depth = 1
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j:k]
    return None


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\text\{\s*([^{}]*)\}", r"\1", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("dfrac", "frac"), ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""),
                 ("$", ""), ("\\$", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s).rstrip(".")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"^(?:x|y|a|b|n|k)=", "", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _num(s: str):
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def math_correct(text: str, gold_solution: str) -> bool:
    x, g = last_boxed(text), last_boxed(gold_solution)
    if x is None or g is None:
        return False
    x, g = normalize(x), normalize(g)
    if x == g:
        return True
    nx, ng = _num(x), _num(g)
    return nx is not None and ng is not None and abs(nx - ng) < 1e-6


if __name__ == "__main__":
    assert math_correct("so \\boxed{\\dfrac{1}{2}}", "is $\\boxed{\\frac12}$.")
    assert math_correct("\\boxed{0.5}", "\\boxed{\\frac{1}{2}}")
    assert math_correct("\\boxed{x=3}", "\\boxed{3}")
    assert math_correct("\\boxed{10^\\circ}", "\\boxed{10}")
    assert not math_correct("\\boxed{4}", "\\boxed{3}")
    assert not math_correct("no box 3", "\\boxed{3}")
    print("math_check ok")
