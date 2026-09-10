"""EXP-C5: 零信号对照 —— 我们的评估协议本身会不会凭空造出归因能力?

一直没跑过的自查. 前面所有归因数字都依赖同一套协议(按题目划分 + 逻辑回归 + 词法特征),
若该协议在**本来没有密钥信号**的数据上也能给出高于随机的准确率, 那么全部结论作废.

两个互补的 null:

  A. clean 轨迹随机分组
     取无 trigger 的 clean 轨迹, 随机贴 K 个标签. 由构造不含任何密钥信息,
     准确率必须落在随机水平. 这检验的是"特征维度高 + 分类器强"会不会自己造出信号.

  B. 标签置换检验(更严格)
     在**真实 triggered 轨迹**上把密钥标签整体置换后重训. 内容、长度、风格分布全部
     与真实实验一致, 唯一被破坏的是"轨迹-密钥"对应关系. 若置换后仍高于随机,
     说明协议在借助与密钥无关的结构(例如题目泄漏)作弊.

判据: 两个 null 的 95% 区间都应覆盖 1/K, 且真实准确率应远在其外.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from baselines import eval_fitted_tfidf, folds, load_traces  # noqa: E402

TFIDF_KW = dict(analyzer="word", ngram_range=(1, 2), max_features=20000)


def load_clean(run: pathlib.Path):
    texts, qid = [], []
    for line in open(run / "clean.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["text"].strip():
            texts.append(r["text"]); qid.append(r["qid"])
    return texts, np.array(qid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/attribution")
    ap.add_argument("--space", default="v2_diverse")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    K = len(json.loads((run / "key_spaces.json").read_text())[args.space])
    chance = 1.0 / K
    rng = np.random.default_rng(0)

    texts, y, qid = load_traces(run, args.space, K)
    fl = folds(qid, args.n_seeds)
    real, _ = eval_fitted_tfidf(texts, y, fl, TFIDF_KW)
    print(f"[real] 真实标签 top-1 = {np.mean(real):.4f} (随机 {chance:.4f})\n", flush=True)

    # --- Null A: clean 轨迹随机分组 ---
    ct, cq = load_clean(run)
    accs_a = []
    for t in range(args.n_perm):
        ya = rng.integers(0, K, size=len(ct))
        a, _ = eval_fitted_tfidf(ct, ya, folds(cq, args.n_seeds), TFIDF_KW)
        if a:
            accs_a.append(float(np.mean(a)))
        print(f"  [nullA] {t+1}/{args.n_perm}", end="\r", flush=True)
    # --- Null B: triggered 轨迹标签置换 ---
    accs_b = []
    for t in range(args.n_perm):
        yb = rng.permutation(y)
        b, _ = eval_fitted_tfidf(texts, yb, fl, TFIDF_KW)
        if b:
            accs_b.append(float(np.mean(b)))
        print(f"  [nullB] {t+1}/{args.n_perm}", end="\r", flush=True)

    print(" " * 30, end="\r")
    out = {"config": vars(args), "chance": chance, "real": float(np.mean(real))}
    ok = True
    for name, accs in [("A: clean 随机分组", accs_a), ("B: 标签置换", accs_b)]:
        m, s = float(np.mean(accs)), float(np.std(accs))
        lo, hi = float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))
        covers = lo <= chance <= hi
        ok &= covers
        out[name[0]] = {"mean": m, "sd": s, "lo": lo, "hi": hi, "covers_chance": covers}
        print(f"[null {name:<16}] {m:.4f} ± {s:.4f}  95%区间 [{lo:.4f}, {hi:.4f}]  "
              f"{'覆盖随机 ✓' if covers else '**未覆盖随机 ✗**'}")

    print("-" * 72)
    # 结论与实测数值挂钩, 不得无条件打印
    if ok and np.mean(real) > max(out["A"]["hi"], out["B"]["hi"]):
        print("=> 两个 null 都落在随机水平, 且真实准确率在其外: 协议不自造信号, 归因结论成立.")
    elif not ok:
        print("=> **有 null 未覆盖随机: 评估协议自身在造信号, 前面所有归因数字须重新审查.**")
    else:
        print("=> null 正常但真实准确率未显著超出 null 区间: 该配置下的归因主张不成立.")

    (run / f"null_control_{args.space}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] {run}/null_control_{args.space}.json")


if __name__ == "__main__":
    main()
