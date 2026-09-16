"""M9 adaptive-distiller scoring (.pipeline/docs/m9_design.md, pre-registered c5775cc). Runs in py312.

Owner tests are M7's, unchanged: the owner's reference students and read-outs are the non-adaptive ones.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
from run_m7 import D, TEST_SEEDS, jl  # noqa: E402
from run_m9 import CONDS, OWNERS, SEEDS  # noqa: E402
from score_m8 import Cell as BaseCell  # noqa: E402
from score_m7 import accuracy  # noqa: E402
from utility_check import extract_answer  # noqa: E402

FAMS = ["qwen15", "llama1b"]


class Cell(BaseCell):
    def load_extra(self):
        out = {}
        for owner in OWNERS:
            for cond in CONDS:
                for s in SEEDS:
                    rows = jl(self.d / f"probe_{self.fam}_grid_{cond}_{owner}_s{s}.jsonl")
                    if rows:
                        out[(cond, owner, s)] = [r["text"] for r in rows]
        return out


def rewrite_check():
    """Manipulation check on the rewritten corpora: answer preserved >= 0.9, length ratio in [0.5, 2]."""
    d = D("gsm")
    info, void = {}, set()
    for owner in OWNERS:
        for cond in CONDS:
            rows = jl(d / f"teacher_{owner}_{cond}.jsonl")
            if not rows:
                continue
            same = np.mean([extract_answer(r["text"]) is not None
                            and extract_answer(r["orig_text"]) is not None
                            and abs(extract_answer(r["text"]) - extract_answer(r["orig_text"])) < 1e-6 for r in rows])
            ratio = np.mean([len(r["text"]) for r in rows]) / np.mean([len(r["orig_text"]) for r in rows])
            acc = accuracy("gsm", rows).mean()
            info[f"{owner}/{cond}"] = {"answer_preserved": float(same), "length_ratio": float(ratio),
                                       "rewritten_acc": float(acc), "truncated": float(np.mean([r["truncated"] for r in rows]))}
            if same < 0.9 or not 0.5 <= ratio <= 2.0:
                void.add(f"{owner}/{cond}")
    return void, info


def main():
    void, info = rewrite_check()
    print("[m9] rewrite check: " + json.dumps(info, indent=0) + f"\n[m9] void corpora: {sorted(void)}", flush=True)
    out = {"rewrite_check": info, "void": sorted(void)}
    res = {}
    for fam in FAMS:
        C = Cell("gsm", fam)
        chk = np.mean([C.rate(C.keys(a, TEST_SEEDS), a, "T1") for a in C.avail])
        print(f"\n[m9/{fam}] T1 TPR check on unattacked M7 students (must be 1.0): {chk}", flush=True)
        for owner, rel in OWNERS.items():
            for cond in CONDS:
                ks = [k for k in C.mix if k[0] == cond and k[1] == owner]
                if not ks:
                    continue
                r = {"n": len(ks), "tpr_T1": C.rate(ks, owner, "T1"), "tpr_T0": C.rate(ks, owner, "T0"),
                     "spoof_T1": C.rate(ks, rel, "T1"), "spoof_T0": C.rate(ks, rel, "T0"),
                     "acc": float(np.mean([accuracy("gsm", jl(D("gsm") / f"probe_{fam}_grid_{k[0]}_{k[1]}_s{k[2]}.jsonl")).mean()
                                           for k in ks]))}
                r["void"] = f"{owner}/{cond}" in void
                res[f"{fam}|{owner}|{cond}"] = r
                print(f"  [m9/{fam}] {owner:<12} {cond}{' [VOID]' if r['void'] else '      '}  T1 TPR {r['tpr_T1']}  spoof(->{rel}) {r['spoof_T1']}  "
                      f"| T0 TPR {r['tpr_T0']} spoof {r['spoof_T0']} | student acc {r['acc']:.3f}", flush=True)
    out["results"] = res
    # ---------------- gates
    gates = {}
    for cond, name in [("ad1", "G1"), ("ad2", "G2")]:
        per_fam = []
        for fam in FAMS:
            v = [res[k]["tpr_T1"] for k in res if k.startswith(f"{fam}|") and k.endswith(f"|{cond}")
                 and res[k]["tpr_T1"] is not None and not res[k]["void"]]
            per_fam.append(np.mean(v) if v else None)
        gates[name] = {"mean_tpr_per_family": per_fam,
                       "pass": bool(all(v is not None and v >= 0.8 for v in per_fam))}
    sp = []
    for fam in FAMS:
        v = [res[k]["spoof_T1"] for k in res if k.startswith(f"{fam}|") and k.endswith("|ad2")
             and res[k]["spoof_T1"] is not None and not res[k]["void"]]
        sp.append(np.mean(v) if v else None)
    gates["n_void_corpora"] = len(void)
    gates["G3"] = {"mean_spoof_per_family": sp, "pass": bool(all(v is not None and v <= 0.3 for v in sp))}
    concl = []
    if not gates["G1"]["pass"]:
        concl.append("G1 FAILS: paraphrase defeats T1 -> primary limitation, scope to non-adaptive distillers")
    if gates["G1"]["pass"] and not gates["G2"]["pass"]:
        concl.append("G2 fails: survives generic rewriting but not targeted imitation")
    if not gates["G3"]["pass"]:
        concl.append("G3 FAILS: ambiguity attack -- attribution can be shifted to an innocent relative")
    if all(gates[g]["pass"] for g in ("G1", "G2", "G3")):
        concl.append("T1 robust to both attacks under a non-adaptive owner")
    out["gates"] = gates; out["conclusion"] = concl
    print("\n== GATES ==\n" + json.dumps(gates, indent=1, default=float) + "\n=> " + "\n=> ".join(concl))
    (D("gsm") / "m9_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
