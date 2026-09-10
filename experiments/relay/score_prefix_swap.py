"""EXP-N2 打分: 续写跟密钥走, 还是跟前缀走?

读出器训练在**原始轨迹尾部**(第 m 步之后), 按题目划分; 测试在续写上, 且续写
**单独打分不含前缀** —— 否则前缀自带的指纹平凡泄漏答案.

对每个条件报两个准确率:
  acc_key    续写被归到**密钥**身份的比例      (nokey 条件下无定义)
  acc_prefix 续写被归到**前缀来源**身份的比例

matched 条件下两者恒等, 作为该配置的天花板.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_and_relay import read_jsonl  # noqa: E402

TFIDF_KW = dict(analyzer="word", ngram_range=(1, 2), max_features=20000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.3)
    args = ap.parse_args()

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    run = pathlib.Path(args.run_dir)
    train = [r for r in (read_jsonl(run / "train_tails.jsonl") or []) if r["text"].strip()]
    K = len({r["key_idx"] for r in train})
    chance = 1.0 / K
    conds = {c: [r for r in (read_jsonl(run / f"{c}.jsonl") or []) if r["text"].strip()]
             for c in ("matched", "swapped", "nokey")}
    print(f"[info] 训练尾部 {len(train)} 条, K={K}, 随机 {chance:.4f}")
    for c, rows in conds.items():
        print(f"       {c}: {len(rows)} 条续写")

    tr_q = np.array([r["qid"] for r in train])
    uq = np.unique(tr_q)
    rows_out = []

    for c, rows in conds.items():
        if not rows:
            continue
        te_q = np.array([r["qid"] for r in rows])
        y_key = np.array([-1 if r["key_idx"] is None else r["key_idx"] for r in rows])
        y_pre = np.array([r["prefix_idx"] for r in rows])
        accs_k, accs_p = [], []
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed)
            u = uq.copy(); rng.shuffle(u)
            hold = set(u[: max(1, int(len(u) * args.test_frac))].tolist())
            tr_m = np.array([q not in hold for q in tr_q])
            te_m = np.array([q in hold for q in te_q])
            if tr_m.sum() < K or te_m.sum() < K:
                continue
            txt_tr = [train[i]["text"] for i in np.where(tr_m)[0]]
            ytr = np.array([train[i]["key_idx"] for i in np.where(tr_m)[0]])
            if len(set(ytr.tolist())) < 2:
                continue
            vec = TfidfVectorizer(**TFIDF_KW).fit(txt_tr)
            Mtr = vec.transform(txt_tr)
            k = min(256, Mtr.shape[1] - 1, Mtr.shape[0] - 1)
            svd = TruncatedSVD(n_components=k, random_state=0).fit(Mtr)
            sc = StandardScaler().fit(svd.transform(Mtr))
            clf = LogisticRegression(max_iter=5000).fit(sc.transform(svd.transform(Mtr)), ytr)

            idx = np.where(te_m)[0]
            Xte = sc.transform(svd.transform(vec.transform([rows[i]["text"] for i in idx])))
            pred = clf.predict(Xte)
            if y_key[idx][0] != -1:
                accs_k.append(float((pred == y_key[idx]).mean()))
            accs_p.append(float((pred == y_pre[idx]).mean()))

        row = {"cond": c, "chance": chance,
               "acc_key": float(np.mean(accs_k)) if accs_k else None,
               "acc_key_sd": float(np.std(accs_k)) if accs_k else None,
               "acc_prefix": float(np.mean(accs_p)) if accs_p else None,
               "acc_prefix_sd": float(np.std(accs_p)) if accs_p else None}
        rows_out.append(row)
        ak = f"{row['acc_key']:.4f}" if row["acc_key"] is not None else "  n/a "
        akx = f"{row['acc_key']/chance:.2f}x" if row["acc_key"] is not None else "  -  "
        print(f"  {c:<9} acc_key={ak} ({akx})   "
              f"acc_prefix={row['acc_prefix']:.4f} ({row['acc_prefix']/chance:.2f}x)")

    # 解释与实测数值挂钩, 不得无条件打印
    d = {r["cond"]: r for r in rows_out}
    print("-" * 74)
    if "swapped" in d and "nokey" in d:
        sk, sp = d["swapped"]["acc_key"], d["swapped"]["acc_prefix"]
        nk = d["nokey"]["acc_prefix"]
        near_chance = nk < chance + 2 * (d["nokey"]["acc_prefix_sd"] or 0)
        if sk is not None and sk > sp * 1.5 and near_chance:
            print("=> **持续条件化**: 冲突时续写跟密钥走, 且去掉密钥后归因塌到随机.")
        elif sp > (sk or 0) * 1.5 and not near_chance:
            print("=> **前缀自维持**: 冲突时续写跟前缀走, 且去掉密钥后仍可归因.")
        elif not near_chance:
            print(f"=> **两者并存**: swapped 下 key={sk:.4f} / prefix={sp:.4f}, "
                  f"nokey 仍高于随机({nk:.4f}) —— 须报告相对量级, 不可单边下结论.")
        else:
            print(f"=> 混合且 nokey 接近随机: key={sk:.4f} / prefix={sp:.4f} —— "
                  "前缀不足以独立维持, 但冲突下的归属需按数值如实描述.")

    (run / "prefix_swap_scores.json").write_text(json.dumps(
        {"config": vars(args), "rows": rows_out}, indent=2))
    print(f"\n[done] {run}/prefix_swap_scores.json")


if __name__ == "__main__":
    main()
