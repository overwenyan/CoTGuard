"""SVRA CPU verifier (p2_design.md §3.2): extract committed arithmetic from a trace and check it.

V1 arithmetic  - every extracted equation re-computes
V2 grounding   - every operand is a problem number, a unit constant, or an earlier result
V4 closure     - the final answer is produced by the extracted computation (or is a problem number)
V3 route       - structural signatures for the CPU-checkable routes (§3.1)

No LLM, no lexical route readout: every decision is a function of extracted numerals and their
order, which is what Prop B (injection immunity) requires.
"""

from __future__ import annotations

import ast
import operator
import re

from utility_check import extract_answer
from verify_redundant import analyze_trace

NUMWORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000, "half": 0.5, "twice": 2, "double": 2,
    "triple": 3, "dozen": 12, "quarter": 0.25, "third": 3, "once": 1,
}
# unit conversions and universal multipliers only; small integers are NOT free
CONSTANTS = {0.0, 1.0, 2.0, 0.5, 7.0, 10.0, 12.0, 24.0, 52.0, 60.0, 100.0, 365.0, 1000.0}

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: lambda a, b: a ** b if abs(b) <= 12 else float("nan")}


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, 1e-3 * abs(b))


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("unsupported")


LABEL_NUM = re.compile(r"\b(?:step|approach|solution|route|method|part|case|month|day|week|option|"
                       r"stage|phase|year|customer|person)\s+\d+\b", re.I)


def _normalize(s: str) -> str:
    s = re.sub(r"(\d{1,3}(?:,\d{3})+)", lambda m: m.group(1).replace(",", ""), s)
    s = s.replace("**", " ")
    s = re.sub(r"\^\s*\{?\s*(\d+)\s*\}?", r" ** \1", s)      # 2^2, 2^{3}
    s = s.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\times|\\cdot|×|·|∗", "*", s)
    s = re.sub(r"\\div|÷", "/", s)
    s = re.sub(r"\([^()]*[A-Za-z]{2,}[^()]*\)", " ", s)   # prose asides: "(from the first 3 customers)"
    s = LABEL_NUM.sub(" ", s)                                  # labels, not quantities: "Month 2", "Step 3"
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\$?\d)", " * ", s)
    s = re.sub(r"(?<=\w)\s+[xX]\s+(?=\$?\d)", " * ", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*(?:of\b)?", r"(\1/100) * ", s)
    s = s.replace("$", " ").replace("\\", " ")
    s = re.sub(r"/\s*[A-Za-z][A-Za-z]+", " ", s)          # per-unit: "$2/egg" -> "2"
    return s


_TOKEN = re.compile(r"\d+(?:\.\d+)?|\*\*|[+\-*/()]|[A-Za-z]+|\S")
_OPTOK = ("+", "-", "*", "/", "(", ")", "**")


def _suffix_expr(lhs: str):
    """Longest suffix of the LHS that is a pure arithmetic expression, words acting as glue."""
    toks = _TOKEN.findall(_normalize(lhs))
    if any(len(t) == 1 and t.isalpha() and t not in ("a", "A", "I") for t in toks[-8:]):
        return None                                        # symbolic (x, y, n): not arithmetic
    toks = [t for t in toks if not t.isalpha()]            # unit words dropped
    best = None
    for start in range(len(toks) - 1, -1, -1):
        cand = toks[start:]
        if any(t not in _OPTOK and not NUM_RE.fullmatch(t) for t in cand):
            break
        # two numbers in a row means prose glued them; the expression cannot extend past that
        if any(NUM_RE.fullmatch(a) and NUM_RE.fullmatch(b) for a, b in zip(cand, cand[1:])):
            break
        expr = " ".join(cand).strip()
        while expr.startswith((")", "*", "/", "+")):
            expr = expr[1:].strip()
        while expr.endswith(("*", "/", "+", "-", "(")):
            expr = expr[:-1].strip()
        nums = NUM_RE.findall(expr)
        if len(nums) >= 2 and re.search(r"\d\s*[+\-*/]|\)\s*[+\-*/]", expr):
            try:
                val = _eval(ast.parse(expr, mode="eval"))
            except Exception:
                continue
            best = (expr, [float(n) for n in re.findall(r"\d+(?:\.\d+)?", expr)],
                    frozenset(re.findall(r"(?<=[\d)\s])[+\-*/](?=[\s\d(])", expr)), val)
    return best


_RHS_NUM = re.compile(r"^\s*(?:\\\(|\\\[|\\\$|\$|\*\*|\s)*\s*(-?\d[\d,]*(?:\.\d+)?)")


_PAREN_DERIV = re.compile(r"(-?\d+(?:\.\d+)?)\s*[A-Za-z ]{0,20}\(\s*([\d.\s+\-*/×]+)\)")
_BARE_RESULT = re.compile(r"^\s*[A-Za-z][A-Za-z ]*=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*[A-Za-z ]*\.?\s*$")


def _eq(expr_src: str, result: float):
    got = _suffix_expr(expr_src)
    if got is None:
        return None
    expr, operands, ops, val = got
    return {"expr": expr, "operands": operands, "ops": ops, "computed": val, "result": result,
            "ok": close(val, result)}


def extract_equations(text: str) -> list[dict]:
    eqs = []
    pending = None                                         # "X = a * b" with the result on the next line
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)   # list bullets are not minus signs
        if pending is not None and line.strip():
            m = _BARE_RESULT.match(line)
            if m:
                e = _eq(pending, float(m.group(1).replace(",", "")))
                if e:
                    eqs.append(e)
                pending = None
                continue
            pending = None
        for m in _PAREN_DERIV.finditer(line):              # "4 hours (5 - 1)": derivation in prose
            if re.search(r"\d\s*[+\-*/×]\s*\d", m.group(2)):
                e = _eq(m.group(2), float(m.group(1)))
                if e and e["ok"]:
                    eqs.append(e)
        parts = line.split("=")
        for i in range(1, len(parts)):
            lhs = parts[i - 1].split(":")[-1]
            m = _RHS_NUM.match(parts[i])
            if not m:
                if i == len(parts) - 1 and _suffix_expr(parts[i]) is not None:
                    pending = parts[i]
                continue
            tail = _normalize(parts[i][m.end():])
            if re.match(r"^\s*[A-Za-z ]{0,20}(?:\*\*|[+\-*/])\s*\d", tail):
                lone = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*", _normalize(lhs))
                if lone and i == len(parts) - 1:           # result-first: "196 = 2 * 98"
                    e = _eq(parts[i], float(lone.group(1)))
                    if e:
                        eqs.append(e)
                elif i == len(parts) - 1 and _suffix_expr(parts[i]) is not None:
                    pending = parts[i]
                continue                                   # RHS continues as an expression
            e = _eq(lhs, float(m.group(1).replace(",", "")))
            if e:
                eqs.append(e)
    return eqs


def problem_numbers(question: str) -> set[float]:
    q = _normalize(question)
    vals = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", q)}
    for w, v in NUMWORDS.items():
        if re.search(rf"\b{w}\b", question, re.I):
            vals.add(float(v))
    for n in re.findall(r"(\d+(?:\.\d+)?)\s*%", question):
        vals.add(float(n) / 100)
    for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", question):
        if float(b):
            vals.add(float(a) / float(b))
    return vals


def _member(v: float, pool) -> bool:
    return any(close(v, p) for p in pool)


def verify(text: str, question: str) -> dict:
    eqs = extract_equations(text)
    ground = problem_numbers(question) | CONSTANTS
    derived: list[float] = []
    v1 = v2 = True
    for e in eqs:
        v1 &= e["ok"]
        v2 &= all(_member(o, ground) or _member(o, derived) for o in e["operands"])
        derived.append(e["result"])
    ans = extract_answer(text)
    v4 = ans is not None and (_member(ans, derived) or _member(ans, problem_numbers(question)))
    has = len(eqs) > 0
    return {"n_eq": len(eqs), "has_eq": has, "v1": has and v1, "v2": has and v2, "v4": has and v4,
            "verified": has and v1 and v2 and v4, "answer": ans, "eqs": eqs}


# ---- V3 route signatures (§3.1); each returns (applicable, passes) ----

def sig_twice(text: str, eqs, k: int = 3):
    return True, analyze_trace(text, 2)["n_agree_pairs"] >= k


def sig_add_first(text: str, eqs):
    kinds = ["add" if e["ops"] == {"+"} else "mul" if ("*" in e["ops"]) else "other" for e in eqs]
    if "add" not in kinds or "mul" not in kinds:
        return False, True                                 # vacuous
    first_mul = kinds.index("mul")
    return True, "add" not in kinds[first_mul + 1:]


def sig_largest_first(text: str, eqs):
    r = [e["result"] for e in eqs]
    if len(r) < 2:
        return False, True
    return True, all(a >= b - 1e-9 for a, b in zip(r, r[1:]))


def sig_fewest_deps(text: str, eqs, question: str = ""):
    if not eqs:
        return False, True
    ground = problem_numbers(question) | CONSTANTS
    return True, all(_member(o, ground) for o in eqs[0]["operands"])


SIGNATURES = {"twice": sig_twice, "add_first": sig_add_first, "largest_first": sig_largest_first}


def node_signature(e: dict):
    return (tuple(sorted(round(o, 4) for o in e["operands"])), tuple(sorted(e["ops"])),
            round(e["result"], 4))


def _selftest():
    q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with "
         "four. She sells the remainder for $2 per egg. How much does she make?")
    good = ("Subtract breakfast: 16 - 3 = 13 eggs.\nThen muffins: 13 - 4 = 9 eggs.\n"
            "Money: 9 eggs * $2/egg = $18.\nTherefore she makes \\boxed{18} dollars.")
    r = verify(good, q)
    assert r["n_eq"] == 3 and r["verified"], r
    bad_arith = good.replace("13 - 4 = 9", "13 - 4 = 8").replace("9 eggs * $2/egg = $18", "8 eggs * $2/egg = $16").replace("{18}", "{16}")
    r = verify(bad_arith, q)
    assert r["n_eq"] == 3 and not r["v1"] and r["v2"] and not r["verified"], r
    invented = good.replace("13 - 4 = 9", "13 - 5 = 8").replace("9 eggs", "8 eggs").replace("$18", "$16").replace("{18}", "{16}")
    r = verify(invented, q)
    assert r["v1"] and not r["v2"] and not r["verified"], r   # 5 is not in the problem
    semantic = good.replace("13 - 4 = 9", "13 + 4 = 17").replace("9 eggs", "17 eggs").replace("$18", "$34").replace("{18}", "{34}")
    r = verify(semantic, q)
    assert r["verified"], r                                     # wrong op, consistent: passes (Prop A limit)
    unclosed = good.replace("\\boxed{18}", "\\boxed{20}")
    assert not verify(unclosed, q)["v4"]
    symbolic = "Let x + y = 16 and 5x + 3y = 64, so the answer is \\boxed{64}."
    assert verify(symbolic, "costs $5 and $3, 16 glasses, pay 64")["n_eq"] == 0
    glued = "She has 5 apples and buys 3 more, so 5 + 3 = 8 apples."
    e = extract_equations(glued)
    assert len(e) == 1 and e[0]["expr"] == "5 + 3", e
    pct = "The discount is 60% of $5 = $3."
    e = extract_equations(pct)
    assert len(e) == 1 and e[0]["ok"], e
    times = "3 sprints x 60 meters = 180 meters, and 3 sessions × 180 = 540."
    e = extract_equations(times)
    assert [x["result"] for x in e] == [180.0, 540.0] and all(x["ok"] for x in e), e
    chain = "Total = 16 - 3 - 4 = 9 eggs"
    e = extract_equations(chain)
    assert len(e) == 1 and e[0]["result"] == 9.0 and e[0]["ok"], e
    thousands = "Cost is 1,200 * 3 = 3,600 dollars"
    e = extract_equations(thousands)
    assert len(e) == 1 and e[0]["ok"] and e[0]["result"] == 3600.0, e
    rhs_expr = "Weekly = 3 * 60 = 180"
    e = extract_equations(rhs_expr)
    assert len(e) == 1 and e[0]["expr"] == "3 * 60", e
    eqs = extract_equations("2 + 3 = 5\n4 + 1 = 5\n5 * 2 = 10")
    assert sig_add_first("", eqs) == (True, True)
    eqs = extract_equations("5 * 2 = 10\n4 + 1 = 5")
    assert sig_add_first("", eqs) == (True, False)
    assert sig_largest_first("", extract_equations("10 * 3 = 30\n4 + 1 = 5")) == (True, True)
    assert sig_largest_first("", extract_equations("4 + 1 = 5\n10 * 3 = 30")) == (True, False)
    e = extract_equations("Total bolts = 2 (blue) + 1 (white) = 3 bolts.")
    assert len(e) == 1 and e[0]["ok"] and e[0]["result"] == 3.0, e
    e = extract_equations("1. **Regular glasses cost:** \\(8 \\times \\$5 = \\$40\\)")
    assert len(e) == 1 and e[0]["ok"] and e[0]["result"] == 40.0, e
    e = extract_equations("**Total:** 40 + 24 = **64**")
    assert len(e) == 1 and e[0]["ok"] and e[0]["result"] == 64.0, e
    e = extract_equations("Total = 3 (from the first 3 customers) + 4 (from the next 2 customers) = 7 DVDs")
    assert len(e) == 1 and e[0]["expr"] == "3 + 4" and e[0]["ok"], e
    e = extract_equations("Month 2 downloads = 60 * 3 = 180")
    assert len(e) == 1 and e[0]["expr"] == "60 * 3" and e[0]["ok"], e
    e = extract_equations("Step 3: Total (Month 1 + Month 2) = 60 + 180 = 240")
    assert [x["result"] for x in e] == [240.0] and e[0]["ok"], e
    e = extract_equations("Since it is from 1:00 PM to 5:00 PM, it represents 4 hours (5 - 1).")
    assert len(e) == 1 and e[0]["expr"] == "5 - 1" and e[0]["result"] == 4.0, e
    e = extract_equations("Centimeters melted = 4 * 2\nCentimeters melted = 8 centimeters")
    assert len(e) == 1 and e[0]["expr"] == "4 * 2" and e[0]["ok"], e
    e = extract_equations("Hours = 5 - 1\nSo the total is large.")
    assert e == [], e
    e = extract_equations("- 6432 ÷ 2 = 3216\n- 3216 ÷ 2 = 1608")
    assert [x["ok"] for x in e] == [True, True], e
    e = extract_equations("3. The next 2 customers buy 2 DVDs each: 2 * 2 = 4 DVDs")
    assert len(e) == 1 and e[0]["expr"] == "2 * 2" and e[0]["ok"], e
    e = extract_equations("196 = 2 * 98\n98 = 2 * 49")
    assert [(x["expr"], x["result"], x["ok"]) for x in e] == [("2 * 98", 196.0, True), ("2 * 49", 98.0, True)], e
    e = extract_equations("So the count is (2 + 1) * (2 + 1) = 9 and 2^2 * 7^2 = 196")
    assert [x["result"] for x in e] == [9.0, 196.0] and all(x["ok"] for x in e), e
    e = extract_equations("x = 5 - 1")
    assert e == [], e
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
