"""M3 v6 dilution threshold curve (m3_design.md v6, pre-registered b53b6f8 / 4028fbb).
Owner test (K=32, lexical read-out, regime-matched vetoes) at N = 1319 (primary) and N = 200."""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("M3C_OUT", str(HERE / "data4"))
import score_m3d as s  # noqa: E402
from run_m3d import CURVE_KEYS  # noqa: E402

D, QS = s.OUT, s.OUT / "qs"
COND = {"dil25": "dil0_clean", "dil50": "dil0_clean", "dil10e3": "dil0e3_clean", "dil10ft": "dil0ft_clean"}
PRED = {"dil25": ">= 2/4", "dil50": ">= 3/4", "dil10e3": "<= 1/4", "dil10ft": "<= 1/4"}


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    X, y, _ = s.teacher(s.E + ["clean"])
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    c = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(X), y)

    def P(name):
        fp = D / f"out_qwen15_{name}.jsonl"
        fp = fp if fp.exists() else QS / f"out_qwen15_{name}.jsonl"
        rows = s.jl(fp)
        return None if rows is None else c.predict_proba(v.transform([r["text"] for r in rows]))[:, : s.K]

    cache = {}
    get = lambda n: cache.setdefault(n, P(n))
    res = {}
    for cond, veto in COND.items():
        res[cond] = {}
        for k in CURVE_KEYS:
            M = get(f"{cond}_{k}")
            if M is None:
                continue
            row = {}
            for N in [200, 1319]:
                idx = np.arange(min(N, len(M)))
                p = s.pval(M[idx].mean(0), k)
                inn = [s.pval(get(vn)[idx].mean(0), k) for vn in [veto, "base"] if get(vn) is not None and len(get(vn)) >= len(idx)]
                row[N] = {"p": p, "innocent_p": inn, "ok": bool(p <= 0.05 and all(q > 0.05 for q in inn) and len(inn) == 2)}
            res[cond][k] = row
        n_ok = sum(res[cond][k][1319]["ok"] for k in res[cond])
        print(f"[v6] {cond:<8} N=1319: {n_ok}/{len(res[cond])} detected (prediction {PRED[cond]})  "
              + "  ".join(f"{k}:p={res[cond][k][1319]['p']:.3f}{'' if res[cond][k][1319]['ok'] else '*'}"
                          f"(N200 {res[cond][k][200]['p']:.3f})" for k in res[cond]), flush=True)
        res[cond]["_detected_1319"] = n_ok
    sys.path.insert(0, str(HERE))
    from run_m3c import correct
    acc = lambda n: float(np.mean([correct("gsm", r["text"], r["gold"]) for r in s.jl(D / f"out_qwen15_{n}.jsonl")[:1319]])) \
        if (D / f"out_qwen15_{n}.jsonl").exists() else None
    util = {n: acc(n) for n in ["dil0e3_clean", "dil0ft_clean"] + [f"{c}_{k}" for c in COND for k in CURVE_KEYS]}
    print("[v6] utility: " + "  ".join(f"{n}:{u:.3f}" for n, u in util.items() if u is not None))
    thr = [c for c in ["dil10e3", "dil10ft", "dil25", "dil50"] if res.get(c, {}).get("_detected_1319", 0) >= 3]
    print(f"[v6] conditions with >= 3/4 detected: {thr or 'none'}")
    (D / "m3f_result.json").write_text(json.dumps({"tests": res, "utility": util}, indent=1, default=str))


if __name__ == "__main__":
    main()
