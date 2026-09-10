"""零 GPU 成本的概念验证: 判别方向能否替代"与指令文本的相似度".

EXP-R0 证明最直觉的统计量(推理步 vs trigger 指令文本的余弦相似度)在控制长度后
检测力等同随机(AUROC 0.543). 诊断是: trigger 的指纹在推理风格里, 不在与指令句的
词汇重叠里.

修正思路利用威胁模型中一个未被使用的优势: **owner 持有密钥, 可生成任意多
(triggered, clean) 配对参考数据**. 于是可以从参考数据里**拟合判别方向**
  w = normalize( mean(emb(triggered steps)) - mean(emb(clean steps)) )
再用 <emb(step), w> 给候选轨迹打分.

本脚本用已有 pilot 数据(24+24 条)做交叉验证, 且**全部评估都 length-matched**,
在投入 GPU 前判断该方向是否值得做.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import auroc, caliper_match, split_steps  # noqa: E402
from trigger import make_trigger  # noqa: E402


def trace_scores(emb_list, w):
    """轨迹分数 = 各步在方向 w 上投影的均值(与长度无关的聚合)."""
    return np.array([float((E @ w).mean()) if len(E) else 0.0 for E in emb_list])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/pilot")
    ap.add_argument("--hop", type=int, default=0)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--true-key", default="patient-teacher-2026")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--caliper", type=int, default=1)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    run = pathlib.Path(args.run_dir)
    recs = [json.loads(l) for l in open(run / f"hop{args.hop}.jsonl") if l.strip()]
    trig = [r for r in recs if r["arm"] == "triggered"]
    clean = [r for r in recs if r["arm"] == "clean"]
    st_t = [split_steps(r["text"]) for r in trig]
    st_c = [split_steps(r["text"]) for r in clean]

    model = SentenceTransformer(args.model)
    print(f"[info] 编码 {len(st_t)} triggered + {len(st_c)} clean 轨迹 ...")
    E_t = [model.encode(s, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False) if s else np.zeros((0, 768)) for s in st_t]
    E_c = [model.encode(s, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False) if s else np.zeros((0, 768)) for s in st_c]

    len_t = np.array([len(s) for s in st_t])
    len_c = np.array([len(s) for s in st_c])

    # --- 参照: 语义匹配范式的"与指令文本相似度" ---
    w_instr = model.encode([make_trigger(args.true_key)], convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)[0]
    a_instr = auroc(trace_scores(E_t, w_instr), trace_scores(E_c, w_instr))

    # --- 判别方向, K 折交叉验证(严格不在训练折上评估) ---
    rng = np.random.default_rng(0)
    idx_t = rng.permutation(len(E_t))
    idx_c = rng.permutation(len(E_c))
    folds_t = np.array_split(idx_t, args.n_folds)
    folds_c = np.array_split(idx_c, args.n_folds)

    s_t_all, s_c_all = np.zeros(len(E_t)), np.zeros(len(E_c))
    for f in range(args.n_folds):
        tr_t = np.concatenate([folds_t[g] for g in range(args.n_folds) if g != f])
        tr_c = np.concatenate([folds_c[g] for g in range(args.n_folds) if g != f])
        mt = np.concatenate([E_t[i] for i in tr_t if len(E_t[i])]).mean(axis=0)
        mc = np.concatenate([E_c[i] for i in tr_c if len(E_c[i])]).mean(axis=0)
        w = mt - mc
        w /= np.linalg.norm(w) + 1e-12
        for i in folds_t[f]:
            s_t_all[i] = float((E_t[i] @ w).mean()) if len(E_t[i]) else 0.0
        for j in folds_c[f]:
            s_c_all[j] = float((E_c[j] @ w).mean()) if len(E_c[j]) else 0.0

    a_disc = auroc(s_t_all, s_c_all)
    a_len = auroc(len_t, len_c)

    # --- length-matched 评估(唯一可信的数字) ---
    pairs = caliper_match(list(len_t), list(len_c), args.caliper)
    if pairs:
        it, ic = zip(*pairs)
        a_instr_lm = auroc(trace_scores([E_t[i] for i in it], w_instr),
                           trace_scores([E_c[j] for j in ic], w_instr))
        a_disc_lm = auroc([s_t_all[i] for i in it], [s_c_all[j] for j in ic])
    else:
        a_instr_lm = a_disc_lm = float("nan")

    print(f"\n{'检测器':<34}{'AUROC':>9}{'AUROC(length-matched)':>24}")
    print("-" * 68)
    print(f"{'len_only (平凡 baseline)':<34}{a_len:>9.4f}{'—':>24}")
    print(f"{'instr-sim (语义匹配基线)':<34}{a_instr:>9.4f}{a_instr_lm:>24.4f}")
    print(f"{'discriminative (本修正)':<34}{a_disc:>9.4f}{a_disc_lm:>24.4f}")
    print(f"\n[note] length-matched 配对数 = {len(pairs)}；"
          f"该列才是控制长度混淆后的真实检测力")

    out = {"config": vars(args), "n_pairs": len(pairs),
           "auroc": {"len_only": a_len, "instr_sim": a_instr, "discriminative": a_disc,
                     "instr_sim_len_matched": a_instr_lm,
                     "discriminative_len_matched": a_disc_lm}}
    (run / f"discriminative_probe_hop{args.hop}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] {run}/discriminative_probe_hop{args.hop}.json")


if __name__ == "__main__":
    main()
