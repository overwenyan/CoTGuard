"""M9b scoring (.pipeline/docs/m9b_design.md). Runs in py312.

Gates G2b/G3b, direction test H-dir (exact paired sign-flip permutation over the 4 adjacent pairs), and the
correlational mechanism analysis H-mech (exact Spearman permutation p).
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "relay"))
from run_m7 import D, OUT, jl  # noqa: E402
from run_m9b import ATTACKS, SEEDS, later, name  # noqa: E402
from score_m8 import Cell as BaseCell  # noqa: E402
from score_m7 import accuracy  # noqa: E402
from utility_check import extract_answer  # noqa: E402

FAMS = ["qwen15", "llama1b"]


class Cell(BaseCell):
    def load_extra(self):
        out = {}
        for owner, target in ATTACKS:
            for s in SEEDS:
                rows = jl(self.d / f"probe_{self.fam}_grid_{name(owner, target)}_s{s}.jsonl")
                if rows:
                    out[(owner, target, s)] = [r["text"] for r in rows]
        return out


def rewrite_check():
    info, void = {}, set()
    for owner, target in ATTACKS:
        rows = jl(D("gsm") / f"teacher_{name(owner, target)}.jsonl")
        if not rows:
            continue
        same = np.mean([(x := extract_answer(r["text"])) is not None and (y := extract_answer(r["orig_text"])) is not None
                        and abs(x - y) < 1e-6 for r in rows])
        ratio = np.mean([len(r["text"]) for r in rows]) / np.mean([len(r["orig_text"]) for r in rows])
        info[f"{owner}->{target}"] = {"answer_preserved": float(same), "length_ratio": float(ratio)}
        if same < 0.9 or not 0.5 <= ratio <= 2.0:
            void.add((owner, target))
    return void, info


def spearman_exact(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None, None
    rho = stats.spearmanr(x, y).correlation
    perms = [stats.spearmanr(x, np.array(p)).correlation for p in itertools.permutations(y)] if len(y) <= 8 else None
    if perms is None:
        return float(rho), None
    p = float(np.mean([abs(r) >= abs(rho) - 1e-12 for r in perms]))
    return float(rho), p


def main():
    void, info = rewrite_check()
    print("[m9b] rewrite check: " + json.dumps(info) + f"\n[m9b] void attacks: {sorted(void)}", flush=True)
    m7 = json.loads((OUT / "m7_result.json").read_text())
    disp = json.loads((D("gsm") / "m9_dispersion.json").read_text())
    out = {"rewrite_check": info, "void": [f"{o}->{t}" for o, t in sorted(void)], "families": {}}
    for fam in FAMS:
        C = Cell("gsm", fam)
        chk = float(np.mean([C.rate(C.keys(a, list(range(10, 20))), a, "T1") for a in C.avail]))
        print(f"\n[m9b/{fam}] T1 TPR check on unattacked M7 students (must be 1.0): {chk}", flush=True)
        auc = m7["gsm"][fam]["auc_test"]
        rows = []
        for owner, target in ATTACKS:
            ks = [k for k in C.mix if k[0] == owner and k[1] == target]
            if not ks:
                continue
            tpr, spoof = C.rate(ks, owner, "T1"), C.rate(ks, target, "T1")
            pair = f"{owner}|{target}" if f"{owner}|{target}" in auc else f"{target}|{owner}"
            r = {"owner": owner, "target": target, "later": later(owner, target), "void": (owner, target) in void,
                 "tpr": tpr, "spoof": spoof, "success": None if tpr is None or spoof is None else 1 - tpr + spoof,
                 "closeness": 1 - auc[pair] if pair in auc else None,
                 "target_narrow": disp[target]["mean_pairwise_cosine"], "owner_narrow": disp[owner]["mean_pairwise_cosine"],
                 "acc": float(np.mean([accuracy("gsm", jl(D("gsm") / f"probe_{fam}_grid_{name(owner, target)}_s{k[2]}.jsonl")).mean()
                                       for k in ks]))}
            rows.append(r)
            print(f"  [m9b/{fam}] {owner:>12} -> {target:<12} {'later  ' if r['later'] else 'earlier'}"
                  f"{' [VOID]' if r['void'] else '       '} TPR {tpr}  spoof {spoof}  success {r['success']}  "
                  f"closeness {r['closeness']:.3f}  acc {r['acc']:.3f}", flush=True)
        valid = [r for r in rows if not r["void"] and r["success"] is not None]
        g2 = float(np.mean([r["tpr"] for r in valid])) if valid else None
        g3 = float(np.mean([r["spoof"] for r in valid])) if valid else None
        mech = {m: dict(zip(["rho", "p_exact"], spearman_exact([r[m] for r in valid], [r["success"] for r in valid])))
                for m in ["closeness", "target_narrow", "owner_narrow"]}
        out["families"][fam] = {"rows": rows, "mean_tpr": g2, "mean_spoof": g3, "mech": mech, "tpr_check": chk}
        print(f"  [m9b/{fam}] valid attacks {len(valid)}/8 | mean TPR {g2} | mean spoof {g3}\n  [m9b/{fam}] H-mech {mech}",
              flush=True)
    # ---------------- gates
    fams = out["families"]
    G2b = all(fams[f]["mean_tpr"] is not None and fams[f]["mean_tpr"] >= 0.8 for f in FAMS)
    G3b = all(fams[f]["mean_spoof"] is not None and fams[f]["mean_spoof"] <= 0.3 for f in FAMS)
    # H-dir: per pair, success(toward later) - success(toward earlier), pooled over families
    diffs = []
    for a, b in [(ATTACKS[k][0], ATTACKS[k][1]) for k in range(0, 8, 2)]:
        lo, hi = (a, b) if later(a, b) else (b, a)                  # lo = earlier stage
        d = []
        for f in FAMS:
            R = {(r["owner"], r["target"]): r for r in fams[f]["rows"] if not r["void"] and r["success"] is not None}
            if (lo, hi) in R and (hi, lo) in R:
                d.append(R[(lo, hi)]["success"] - R[(hi, lo)]["success"])     # toward later minus toward earlier
        if d:
            diffs.append(float(np.mean(d)))
    if diffs:
        obs = np.mean(diffs)
        flips = [np.mean([s * v for s, v in zip(signs, diffs)]) for signs in itertools.product([1, -1], repeat=len(diffs))]
        p_dir = float(np.mean([f >= obs - 1e-12 for f in flips]))
        same_sign = all(v > 0 for v in diffs) or all(v < 0 for v in diffs)
    else:
        obs, p_dir, same_sign = None, None, False
    out["gates"] = {"G2b_pass": G2b, "G3b_pass": G3b,
                    "H_dir": {"per_pair_diff_later_minus_earlier": diffs, "mean": obs, "p_exact_one_sided": p_dir,
                              "all_pairs_same_sign": same_sign, "n_pairs": len(diffs)}}
    concl = []
    concl.append("G2b fails: imitation evasion replicates with valid corpora" if not G2b else
                 "G2b PASSES: M9's evasion is not robust to an answer-preserving rewrite")
    concl.append("G3b fails: spoofing replicates" if not G3b else
                 "G3b PASSES: M9's ambiguity attack is prompt-dependent; withdraw from the abstract")
    concl.append(f"H-dir: mean later-minus-earlier success {obs}, exact one-sided p {p_dir} over {len(diffs)} pairs; "
                 + ("a direction rule may be stated (all pairs agree)" if same_sign and len(diffs) == 4 else
                    "no direction rule may be stated"))
    print("\n== GATES ==\n" + json.dumps(out["gates"], default=float) + "\n=> " + "\n=> ".join(concl))
    out["conclusion"] = concl
    (D("gsm") / "m9b_result.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
