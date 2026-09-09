"""长度混淆对照: 检测是否只是平凡地由轨迹长度驱动.

Pilot (EXP-R0) 发现 trigger 使 GSM8K 轨迹平均步数从 7.38 涨到 14.38 (约 2x).
若一个只数步数的检测器就能达到高 AUROC, 则语义检测的主结果失去意义.

本脚本给出三条曲线:
  len_only     只用步数做判别 (平凡 baseline, 必须报告)
  semantic     语义相似度检测
  semantic_lm  长度匹配子样本上的语义检测 (control for length)

长度匹配用 caliper 匹配: 对每条 triggered 轨迹, 找步数最接近的 clean 轨迹配对,
超出 caliper 的丢弃, 使两臂步数分布对齐后再比较.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from trigger import make_trigger, wrong_keys  # noqa: E402


def split_steps(text):
    parts = [s.strip() for s in text.replace("\n\n", "\n").split("\n")]
    return [p for p in parts if len(p) > 5]


def auroc(pos, neg):
    """Mann-Whitney U 的 AUROC, 处理并列."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    # 并列取平均秩
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    rp = r[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def caliper_match(len_t, len_c, caliper=1):
    """为每条 triggered 找步数差 <= caliper 的 clean 配对(不放回)."""
    used = set()
    pairs = []
    order_c = sorted(range(len(len_c)), key=lambda i: len_c[i])
    for i, lt in enumerate(len_t):
        best, bd = None, None
        for j in order_c:
            if j in used:
                continue
            d = abs(len_c[j] - lt)
            if bd is None or d < bd:
                best, bd = j, d
            if d == 0:
                break
        if best is not None and bd is not None and bd <= caliper:
            used.add(best)
            pairs.append((i, best))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--hop", type=int, default=0)
    ap.add_argument("--n-wrong", type=int, default=127)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--true-key", default="patient-teacher-2026")
    ap.add_argument("--caliper", type=int, default=1)
    ap.add_argument("--n-calib-problems", type=int, default=1500)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    recs = [json.loads(l) for l in open(run / f"hop{args.hop}.jsonl") if l.strip()]
    trig = [r for r in recs if r["arm"] == "triggered"]
    clean = [r for r in recs if r["arm"] == "clean"]
    st_t = [split_steps(r["text"]) for r in trig]
    st_c = [split_steps(r["text"]) for r in clean]
    len_t = [len(s) for s in st_t]
    len_c = [len(s) for s in st_c]
    print(f"[len] triggered mean={np.mean(len_t):.2f} sd={np.std(len_t):.2f}  "
          f"clean mean={np.mean(len_c):.2f} sd={np.std(len_c):.2f}")

    # --- 平凡 baseline: 只用步数 ---
    a_len = auroc(len_t, len_c)
    print(f"[len_only]  AUROC = {a_len:.4f}   <-- 若接近 1, 检测被长度平凡驱动")

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    taus = [make_trigger(args.true_key)] + wrong_keys(args.true_key, args.n_wrong)
    model = SentenceTransformer(args.model)
    e_tau = model.encode(taus, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)

    cal = load_dataset("openai/gsm8k", "main", split="train")
    cal = cal.select(range(min(args.n_calib_problems, len(cal))))
    cal_steps = [s for r in cal for s in split_steps(r["answer"])]
    e_cal = model.encode(cal_steps, batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    cal_sorted = np.sort(e_cal @ e_tau.T, axis=0)

    def trace_score(steps_list):
        """轨迹级分数 = 该轨迹各步共形 p 值的 -log 均值 (与长度无关的聚合)."""
        out = []
        for ss in steps_list:
            if not ss:
                out.append(0.0); continue
            sim = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                               show_progress_bar=False) @ e_tau.T
            rs = np.empty_like(sim)
            for j in range(sim.shape[1]):
                rs[:, j] = np.searchsorted(cal_sorted[:, j], sim[:, j], side="left") / cal_sorted.shape[0]
            pv = (1.0 + (rs[:, 1:] >= rs[:, :1]).sum(axis=1)) / len(taus)
            out.append(float(np.mean(-np.log(np.clip(pv, 1e-12, None)))))  # 均值, 非求和
        return out

    print("[score] 计算语义分数 ...")
    s_t, s_c = trace_score(st_t), trace_score(st_c)
    a_sem = auroc(s_t, s_c)
    print(f"[semantic]  AUROC = {a_sem:.4f}")

    # --- 长度匹配后的语义检测 ---
    pairs = caliper_match(len_t, len_c, args.caliper)
    if pairs:
        it, ic = zip(*pairs)
        a_lm = auroc([s_t[i] for i in it], [s_c[j] for j in ic])
        lm_lt = [len_t[i] for i in it]; lm_lc = [len_c[j] for j in ic]
        print(f"[matched]   n_pairs={len(pairs)}  步数 triggered={np.mean(lm_lt):.2f} "
              f"clean={np.mean(lm_lc):.2f}  AUROC = {a_lm:.4f}   <-- 控制长度后的真实语义检测力")
    else:
        a_lm = float("nan")
        print("[matched]   无可配对样本 (两臂长度分布几乎不重叠)")

    res = {"config": vars(args),
           "len": {"trig_mean": float(np.mean(len_t)), "clean_mean": float(np.mean(len_c)),
                   "trig_sd": float(np.std(len_t)), "clean_sd": float(np.std(len_c))},
           "auroc": {"len_only": a_len, "semantic": a_sem, "semantic_len_matched": a_lm},
           "n_pairs": len(pairs)}
    (run / f"length_control_hop{args.hop}.json").write_text(json.dumps(res, indent=2))
    print(f"[done] {run}/length_control_hop{args.hop}.json")


if __name__ == "__main__":
    main()
