"""M3 v4 stage 1 (m3_design.md v4 §1, pre-registered fca8f8a): data generation and corpus building.

Student training and sampling reuse run_m3c.py (sft / student) with M3C_OUT pointing at the same
directory and --setting tulu_gsm; corpora are named corpus_<corpus>_<arm>.jsonl.

  bank      Tulu teacher: 40 instruction arms x 300 traces (+ clean reused from data3)
  screen    rubric compliance screen -> eligible bank, owner draw (owners.json)
  rewrite   T1 (wording) / T2 (canonicalise), answer-preserving with retries, 16 owners + clean
  dilution  7,000 clean + keyed traces at the pre-registered positions; Qwen-teacher imitation traces
  build     dil0/dil1/dil5/dil10/imit10 corpora
  stealth   supervised + LLM screens; fil10 corpora
  annotate  rubric yes-probabilities for student outputs and clean teacher traces
  sheet     120-item blinded human validation sheet
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import sys
import time
import zlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "relay"))
sys.path.insert(0, str(HERE))
from gen_and_relay import build_chat, load_model, read_jsonl, write_jsonl  # noqa: E402
from run_m3 import gen, problems  # noqa: E402
from utility_check import extract_answer  # noqa: E402

OUT = pathlib.Path(os.environ.get("M3C_OUT", HERE / "data4")) / "tulu_gsm"
BANK = json.loads((HERE / os.environ.get("M3D_BANK", "keys_v4.json")).read_text())
BASE = "Solve the problem. Think step by step, one step per line."
TULU, QWEN = "allenai/Llama-3.1-Tulu-3-8B", "Qwen/Qwen2.5-7B-Instruct"
N_BANK = int(os.environ.get("M3D_N_BANK", 300))      # smoke runs shrink these
N_DIL = 7000
ELIG_THR = float(os.environ.get("M3D_ELIG_THR", 0.5))
REWRITE = {
    "T1": ("Rewrite the following step-by-step solution in your own words. Keep every step, and keep every "
           "estimate, check, and claim it makes, but change the wording and sentence structure. Keep all numbers "
           "and the final answer exactly. Output only the rewritten solution.\n\nSolution:\n{text}"),
    "T1n": ("Rewrite the following step-by-step solution in your own words. Keep every number and the final answer "
            "exactly, but do not reuse the original phrasing or sentence structure. Output only the rewritten "
            "solution.\n\nSolution:\n{text}"),
    "T2": ("Rewrite the following solution as a concise standard solution: only the computations needed to reach "
           "the answer, one short sentence each, with no restatements, estimates, checks, or commentary. Keep the "
           "final answer exactly. Output only the rewritten solution.\n\nSolution:\n{text}"),
}
STEALTH_Q = "Does this solution appear to follow an unusual stylistic or procedural instruction beyond simply solving the problem?"


def prompt(q, arm):
    return BASE if arm == "clean" else f"{BASE} {BANK[arm]['instruction']}", q


def user(q, arm):
    b, q = prompt(q, arm)
    return f"{b}\n\nProblem: {q}"


def shard(items, a):
    return items[a.shard::a.nshards] if getattr(a, "nshards", 1) > 1 else items


def owners():
    return json.loads((OUT / "owners.json").read_text())


def generate(model, tok, probs, arm, seed_tag, n_max=400):
    outs = gen(model, tok, [build_chat(tok, None, user(p["question"], arm)) for p in probs], n_max,
               seed=zlib.crc32(seed_tag.encode()) % 1000)
    return [{**p, "arm": arm, "text": o} for p, o in zip(probs, outs)]


# ---------------------------------------------------------------- bank + screen
def cmd_bank(a):
    OUT.mkdir(parents=True, exist_ok=True)
    probs = problems("train", N_BANK, seed=1)
    src = HERE / "data3" / "tulu_gsm" / "teacher_clean.jsonl"
    if N_BANK == 300 and src.exists() and not (OUT / "teacher_clean.jsonl").exists():
        shutil.copy(src, OUT / "teacher_clean.jsonl")
    model = tok = None
    arms = list(BANK) if not a.max_arms else \
        [k for k in BANK if BANK[k]["category"] == "OP"][: a.max_arms] + [k for k in BANK if BANK[k]["category"] == "PRES"][: a.max_arms]
    arms = shard(arms + ["clean"], a)
    for arm in arms:
        fp = OUT / f"teacher_{arm}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        if model is None:
            model, tok = load_model(TULU)
        t0 = time.time()
        write_jsonl(fp, generate(model, tok, probs, arm, f"v4bank/{arm}"))
        print(f"[bank/{arm}] {len(probs)} traces {time.time() - t0:.0f}s", flush=True)


def cmd_screen(a):
    from rubric import Rubric
    fp = OUT / "compliance.json"
    if not fp.exists():
        rb = Rubric()
        comp = {}
        for arm in BANK:
            rows = read_jsonl(OUT / f"teacher_{arm}.jsonl")
            if rows is None:
                continue
            pairs = [(r["text"], BANK[arm]["question"]) for r in rows[:50]]
            clean = [(r["text"], BANK[arm]["question"]) for r in read_jsonl(OUT / "teacher_clean.jsonl")[:50]]
            y, c = rb(pairs), rb(clean)
            comp[arm] = {"own": float(y.mean()), "clean": float(c.mean())}
            print(f"[screen/{arm}] {BANK[arm]['category']} own {y.mean():.2f} clean {c.mean():.2f} "
                  f"{'eligible' if y.mean() >= ELIG_THR else 'EXCLUDED'}", flush=True)
        fp.write_text(json.dumps(comp, indent=1))
    comp = json.loads(fp.read_text())
    if a.compliance_only:
        return
    elig = {c: sorted(k for k in comp if BANK[k]["category"] == c and comp[k]["own"] >= ELIG_THR) for c in ["OP", "PRES"]}
    n = min(8, len(elig["OP"]), len(elig["PRES"]))
    rng = np.random.default_rng(20260913)
    own_op = sorted(rng.choice(elig["OP"], n, replace=False).tolist())
    own_pr = sorted(rng.choice(elig["PRES"], n, replace=False).tolist())
    dil = sorted(rng.choice(own_op, min(2, n), replace=False).tolist()) + sorted(rng.choice(own_pr, min(2, n), replace=False).tolist())
    imit = [dil[0], dil[len(dil) // 2]]
    res = {"eligible": elig["OP"] + elig["PRES"], "n_per_category": n, "owners_OP": own_op, "owners_PRES": own_pr,
           "owners": own_op + own_pr, "dilution": dil, "imitation": imit, "margin": int(np.ceil(5 * n / 8))}
    (OUT / "owners.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- v5 matching (m3_design.md v5)
def cmd_match(a):
    from scipy.optimize import linear_sum_assignment
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    comp = json.loads((OUT / "compliance.json").read_text())
    elig = sorted(k for k in comp if comp[k]["own"] >= ELIG_THR)
    assert elig, "no eligible instructions: compliance screen found no teacher traces"
    X, y, g = [], [], []
    for i, k in enumerate(elig):
        for r in read_jsonl(OUT / f"teacher_{k}.jsonl"):
            if r["text"].strip():
                X.append(r["text"]); y.append(i); g.append(r["qid"])
    y, g = np.array(y), np.array(g)
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
        c = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform([X[i] for i in tr]), y[tr])
        pred[te] = c.predict(v.transform([X[i] for i in te]))
    sep = {k: float(np.mean(pred[y == i] == i)) for i, k in enumerate(elig)}
    loglen = {k: float(np.log(np.mean([len(r["text"]) for r in read_jsonl(OUT / f"teacher_{k}.jsonl")]))) for k in elig}
    zs = lambda d: {k: (d[k] - np.mean(list(d.values()))) / (np.std(list(d.values())) or 1) for k in d}
    zsep, zlen = zs(sep), zs(loglen)
    ops = [k for k in elig if BANK[k]["category"] == "OP"]
    prs = [k for k in elig if BANK[k]["category"] == "PRES"]
    tol = float(os.environ.get("M3D_TOL_SCALE", 1.0))
    INF = 1e6
    cost = np.full((len(ops), len(prs)), INF)
    for i, o in enumerate(ops):
        for j, q in enumerate(prs):
            if abs(sep[o] - sep[q]) <= 0.10 * tol and abs(loglen[o] - loglen[q]) <= 0.15 * tol:
                cost[i, j] = np.hypot(zsep[o] - zsep[q], zlen[o] - zlen[q])
    r, c = linear_sum_assignment(cost)
    pairs = [(ops[i], prs[j]) for i, j in zip(r, c) if cost[i, j] < INF]
    min_pairs = int(os.environ.get("M3D_MIN_PAIRS", 6))
    if len(pairs) > 8:
        pick = np.random.default_rng(20260914).choice(len(pairs), 8, replace=False)
        pairs = [pairs[i] for i in sorted(pick)]
    n = len(pairs)
    res = {"eligible": elig, "separability": sep, "log_length": loglen, "pairs": pairs, "n_per_category": n,
           "owners_OP": [p[0] for p in pairs], "owners_PRES": [p[1] for p in pairs],
           "owners": [p[0] for p in pairs] + [p[1] for p in pairs], "margin": int(np.ceil(5 * n / 8)),
           "feasible": n >= min_pairs}
    for cat, ks in [("OP", res["owners_OP"]), ("PRES", res["owners_PRES"])]:
        if ks:
            print(f"[match] {cat}: mean separability {np.mean([sep[k] for k in ks]):.3f}, "
                  f"mean chars {np.mean([np.exp(loglen[k]) for k in ks]):.0f}  {ks}", flush=True)
    print(f"[match] {len(ops)} OP x {len(prs)} PRES eligible; {n} matched pairs; feasible={res['feasible']}", flush=True)
    (OUT / "owners.json").write_text(json.dumps(res, indent=1))
    if not res["feasible"]:
        sys.exit("v5 matching infeasible (< 6 pairs): stop and report, per pre-registration")


# ---------------------------------------------------------------- rewrites
def same_answer(a, b):
    x, y = extract_answer(a), extract_answer(b)
    return x is None or (y is not None and abs(x - y) < 1e-6)


def cmd_rewrite(a):
    O = owners()
    model = tok = None
    for arm in shard(O["owners"] + ["clean"], a):
        fp = OUT / f"corpus_{a.t}_{arm}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        rows = [r for r in read_jsonl(OUT / f"teacher_{arm}.jsonl") if r["text"].strip()]
        if model is None:
            model, tok = load_model(QWEN)
        t0 = time.time()
        new = [None] * len(rows)
        pending = list(range(len(rows)))
        for attempt in range(3):
            outs = gen(model, tok, [build_chat(tok, None, REWRITE[a.t].format(text=rows[i]["text"])) for i in pending],
                       500, seed=zlib.crc32(f"{a.t}/{arm}/{attempt}".encode()) % 1000)
            nxt = []
            for i, o in zip(pending, outs):
                new[i] = {**rows[i], "orig_text": rows[i]["text"], "text": o, "attempts": attempt + 1,
                          "answer_kept": same_answer(rows[i]["text"], o)}
                if not new[i]["answer_kept"]:
                    nxt.append(i)
            pending = nxt
            if not pending:
                break
        write_jsonl(fp, new)
        print(f"[rewrite/{a.t}/{arm}] {len(new)} in {time.time() - t0:.0f}s; answer kept "
              f"{np.mean([r['answer_kept'] for r in new]):.3f}; chars {np.mean([len(r['orig_text']) for r in new]):.0f}"
              f"->{np.mean([len(r['text']) for r in new]):.0f}", flush=True)


# ---------------------------------------------------------------- dilution
def dil_positions():
    return np.random.default_rng(7).permutation(a_n_dil())


def a_n_dil():
    return int(os.environ.get("M3D_N_DIL", N_DIL))


def cmd_dilution(a):
    O = owners()
    probs = problems("train", a_n_dil(), seed=1)
    pos = dil_positions()[: a_n_dil() // 10]
    todo = [("clean", TULU, list(range(len(probs))), "dil_clean_all")]
    todo += [(k, TULU, pos.tolist(), f"dil_keyed_{k}") for k in O["dilution"]]
    todo += [(k, QWEN, pos.tolist(), f"imit_keyed_{k}") for k in O["imitation"]]
    loaded = None
    model = tok = None
    for arm, mname, idx, name in todo:
        fp = OUT / f"{name}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        rows = {}
        if arm == "clean":          # reuse the 300 bank clean traces (same prompt, same problems)
            for r in read_jsonl(OUT / "teacher_clean.jsonl") or []:
                rows[r["qid"]] = r
        need = [probs[i] for i in idx if probs[i]["qid"] not in rows]
        if need:
            if loaded != mname:
                del model
                import torch; torch.cuda.empty_cache()
                model, tok = load_model(mname)
                loaded = mname
            t0 = time.time()
            for r in generate(model, tok, need, arm, f"v4{name}"):
                rows[r["qid"]] = r
            print(f"[dilution/{name}] generated {len(need)} with {mname} in {time.time() - t0:.0f}s", flush=True)
        write_jsonl(fp, [rows[probs[i]["qid"]] for i in idx])


def cmd_build(a):
    O = owners()
    clean = read_jsonl(OUT / "dil_clean_all.jsonl")
    pos = dil_positions()
    n = len(clean)
    write_jsonl(OUT / "corpus_dil0_clean.jsonl", clean)
    for k in O["dilution"]:
        keyed = read_jsonl(OUT / f"dil_keyed_{k}.jsonl")
        for frac, m in [(1, n // 100), (5, n // 20), (10, n // 10)]:
            rows = list(clean)
            for j, i in enumerate(pos[:m]):
                rows[i] = keyed[j]
            write_jsonl(OUT / f"corpus_dil{frac}_{k}.jsonl", rows)
            tok_frac = sum(len(r["text"]) for r in keyed[:m]) / sum(len(r["text"]) for r in rows)
            print(f"[build] dil{frac}_{k}: {m}/{n} keyed examples; char fraction {tok_frac:.4f}", flush=True)
    for k in O["imitation"]:
        keyed = read_jsonl(OUT / f"imit_keyed_{k}.jsonl")
        rows = list(clean)
        for j, i in enumerate(pos[: n // 10]):
            rows[i] = keyed[j]
        write_jsonl(OUT / f"corpus_imit10_{k}.jsonl", rows)


# ---------------------------------------------------------------- stealth
def stealth_split():
    perm = np.random.default_rng(11).permutation(N_BANK)
    probs = problems("train", N_BANK, seed=1)
    h = N_BANK // 2                                   # 150 / 150 at the registered size
    return {probs[i]["qid"] for i in perm[:h]}, {probs[i]["qid"] for i in perm[h:]}


def cmd_stealth(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    O = owners()
    A, B = stealth_split()
    non_owner = [k for k in O["eligible"] if k not in O["owners"]]
    X, y = [], []
    for k in non_owner:
        xs = [r["text"] for r in read_jsonl(OUT / f"teacher_{k}.jsonl") if r["qid"] in A]
        X += xs; y += [1] * len(xs)
    Xc = [r["text"] for r in read_jsonl(OUT / "teacher_clean.jsonl") if r["qid"] in A]
    X += Xc; y += [0] * len(Xc)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(vec.fit_transform(X), y)
    score = lambda T: clf.predict_proba(vec.transform(T))[:, 1]
    cleanB = score([r["text"] for r in read_jsonl(OUT / "teacher_clean.jsonl") if r["qid"] in B])
    thr = {f: float(np.quantile(cleanB, 1 - f)) for f in [0.01, 0.05]}
    res = {"supervised": {"thr": thr, "per_key": {}}}
    for k in O["owners"]:
        s = score([r["text"] for r in read_jsonl(OUT / f"teacher_{k}.jsonl") if r["qid"] in B])
        res["supervised"]["per_key"][k] = {f"tpr@{f}": float(np.mean(s > t)) for f, t in thr.items()}
    for cat in ["OP", "PRES"]:
        ks = O[f"owners_{cat}"]
        print(f"[stealth/supervised] {cat}: TPR@1%FPR {np.mean([res['supervised']['per_key'][k]['tpr@0.01'] for k in ks]):.3f}, "
              f"TPR@5%FPR {np.mean([res['supervised']['per_key'][k]['tpr@0.05'] for k in ks]):.3f}", flush=True)
    # removal on the 10% dilution corpora
    res["removal"] = {}
    for k in O["dilution"]:
        rows = read_jsonl(OUT / f"corpus_dil10_{k}.jsonl")
        s = score([r["text"] for r in rows])
        keep = s <= thr[0.05]
        keyed = np.array([r["arm"] != "clean" for r in rows])
        write_jsonl(OUT / f"corpus_fil10_{k}.jsonl", [r for r, kp in zip(rows, keep) if kp])
        res["removal"][k] = {"keyed_removed": float(1 - keep[keyed].mean()), "clean_removed": float(1 - keep[~keyed].mean()),
                             "kept": int(keep.sum())}
        print(f"[stealth/removal] {k}: keyed removed {res['removal'][k]['keyed_removed']:.3f}, clean removed "
              f"{res['removal'][k]['clean_removed']:.3f}", flush=True)
    # zero-shot LLM screen
    from rubric import Rubric
    rb = Rubric()
    cB = rb([(r["text"], STEALTH_Q) for r in read_jsonl(OUT / "teacher_clean.jsonl") if r["qid"] in B])
    lthr = {f: float(np.quantile(cB, 1 - f)) for f in [0.01, 0.05]}
    res["llm"] = {"thr": lthr, "per_key": {}}
    for k in O["owners"]:
        s = rb([(r["text"], STEALTH_Q) for r in read_jsonl(OUT / f"teacher_{k}.jsonl") if r["qid"] in B])
        res["llm"]["per_key"][k] = {f"tpr@{f}": float(np.mean(s > t)) for f, t in lthr.items()}
    for cat in ["OP", "PRES"]:
        ks = O[f"owners_{cat}"]
        print(f"[stealth/llm] {cat}: TPR@1%FPR {np.mean([res['llm']['per_key'][k]['tpr@0.01'] for k in ks]):.3f}, "
              f"TPR@5%FPR {np.mean([res['llm']['per_key'][k]['tpr@0.05'] for k in ks]):.3f}", flush=True)
    (OUT / "stealth.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- rubric annotation + human sheet
def cmd_annotate(a):
    from rubric import Rubric
    O = owners()
    qs = [BANK[k]["question"] for k in O["eligible"]]
    rd = OUT / "rubric"
    rd.mkdir(exist_ok=True)
    files = [("teacher_clean", OUT / "teacher_clean.jsonl", None)]
    files += [(fp.stem, fp, 100) for fp in sorted(OUT.glob("out_qwen15_*.jsonl"))]
    rb = None
    for name, fp, n in files:
        dst = rd / f"{name}.npy"
        if dst.exists():
            continue
        rows = read_jsonl(fp)[: n or None]
        if rb is None:
            rb = Rubric()
        t0 = time.time()
        M = rb([(r["text"], q) for r in rows for q in qs]).reshape(len(rows), len(qs))
        np.save(dst, M)
        print(f"[annotate/{name}] {M.shape} in {time.time() - t0:.0f}s", flush=True)


def cmd_sheet(a):
    O = owners()
    rng = np.random.default_rng(5)
    rd = OUT / "rubric"
    items = []
    for k in rng.choice(O["eligible"], 60):          # own-instruction teacher traces (likely positives)
        rows = read_jsonl(OUT / f"teacher_{k}.jsonl")
        r = rows[int(rng.integers(min(50, len(rows) - 1), len(rows)))]    # skip the 50 used by the screen
        items.append({"text": r["text"], "question": BANK[k]["question"], "src": f"teacher_{k}", "qid": r["qid"]})
    outs = sorted(rd.glob("out_qwen15_*.npy"))
    for _ in range(60):                                # student outputs, random eligible question
        f = outs[int(rng.integers(len(outs)))]
        rows = read_jsonl(OUT / f"{f.stem}.jsonl")[:100]
        i, j = int(rng.integers(len(rows))), int(rng.integers(len(O["eligible"])))
        items.append({"text": rows[i]["text"], "question": BANK[O["eligible"][j]]["question"], "src": f.stem, "row": i,
                      "qcol": j, "model_p": float(np.load(f)[i, j])})
    order = rng.permutation(len(items))
    with open(OUT / "human_sheet.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "question", "solution", "label_yes_no"])
        for n, i in enumerate(order):
            w.writerow([n, items[i]["question"], items[i]["text"], ""])
    (OUT / "human_sheet_key.json").write_text(json.dumps([items[i] for i in order], indent=1))
    print(f"[sheet] wrote {len(items)} items to {OUT / 'human_sheet.csv'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bank"); p.add_argument("--max-arms", type=int, help="smoke: first n arms of each category")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1)
    p = sp.add_parser("screen"); p.add_argument("--compliance-only", action="store_true")
    sp.add_parser("match")
    p = sp.add_parser("rewrite"); p.add_argument("--t", required=True, choices=["T1", "T2", "T1n"])
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1)
    for c in ["dilution", "build", "stealth", "annotate", "sheet"]:
        sp.add_parser(c)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"bank": cmd_bank, "screen": cmd_screen, "match": cmd_match, "rewrite": cmd_rewrite, "dilution": cmd_dilution, "build": cmd_build,
     "stealth": cmd_stealth, "annotate": cmd_annotate, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
