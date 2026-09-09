"""EXP-R1b: 密钥身份可辨识性的上界估计.

EXP-R1 用"轨迹 vs pattern 文本"的原始相似度做归因, top-1 准确率在三种密钥空间上
都停在随机水平(0.127/0.127/0.137 vs 随机 0.125). 但那是一个弱检测器 —— 审稿人会问
"换个强的呢".

本脚本给出**上界**: 直接问"密钥身份是否以任何形式编码在轨迹里".
不再依赖 pattern 文本, 而是从生成数据里**学**每个密钥的表示:

  nearest_centroid  每个密钥的训练轨迹均值作为原型, 测试轨迹归到最近原型
  logistic          多类逻辑回归 (更强, 能利用判别方向)

严格 train/test 按**题目**划分(而非按轨迹), 避免同题不同密钥的内容泄漏 ——
否则分类器可能靠"这是第几题"而非密钥特征作弊.

若连这个上界都接近随机, 则"语义 CoT trigger 不支持归因"是可靠结论.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from length_control import split_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/attribution")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    spaces = json.loads((run / "key_spaces.json").read_text())

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    # 编码 2400+ 条轨迹很慢, 缓存以便反复分析(换分类器/换种子)
    cache = run / f"trace_vecs_{args.model.split('/')[-1]}.npz"

    def trace_vec(text):
        """轨迹表示 = 各步 embedding 的均值(与长度无关)."""
        ss = split_steps(text)
        if not ss:
            return None
        E = model.encode(ss, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)
        v = E.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    cached = dict(np.load(cache)) if cache.exists() else {}
    if cached:
        print(f"[cache] 复用 {cache.name}")

    results = {}
    for space, entries in spaces.items():
        if f"{space}__X" in cached:
            X, y, qid = (cached[f"{space}__X"], cached[f"{space}__y"], cached[f"{space}__q"])
        else:
            X, y, qid = [], [], []
            for ki in range(len(entries)):
                fp = run / f"{space}__key{ki:02d}.jsonl"
                if not fp.exists():
                    continue
                for r in (json.loads(l) for l in open(fp) if l.strip()):
                    v = trace_vec(r["text"])
                    if v is not None:
                        X.append(v); y.append(ki); qid.append(r["qid"])
            if not X:
                continue
            X = np.array(X); y = np.array(y); qid = np.array(qid)
            cached[f"{space}__X"] = X; cached[f"{space}__y"] = y; cached[f"{space}__q"] = qid
            np.savez_compressed(cache, **cached)
        K = len(set(y.tolist()))

        # 按题目划分, 防止同题内容泄漏
        rng = np.random.default_rng(args.seed)
        uq = np.unique(qid)
        rng.shuffle(uq)
        n_test = max(1, int(len(uq) * args.test_frac))
        test_q = set(uq[:n_test].tolist())
        te = np.array([q in test_q for q in qid])
        tr = ~te

        # --- nearest centroid ---
        cent = np.stack([X[tr & (y == k)].mean(axis=0) for k in range(K)])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        pred_nc = (X[te] @ cent.T).argmax(axis=1)
        acc_nc = float((pred_nc == y[te]).mean())

        # --- 多类逻辑回归 (sklearn>=1.7 已移除 multi_class 参数, 默认即 multinomial) ---
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=5000, C=1.0)
            clf.fit(X[tr], y[tr])
            acc_lr = float(clf.score(X[te], y[te]))
            # top-3: 所有权场景下"缩小到少数嫌疑密钥"也有价值
            proba = clf.predict_proba(X[te])
            top3 = np.argsort(-proba, axis=1)[:, :3]
            acc_lr3 = float(np.mean([y[te][i] in top3[i] for i in range(len(top3))]))
        except Exception as e:
            acc_lr = acc_lr3 = None
            print(f"  [warn] logistic 失败: {e}")

        results[space] = {"n_keys": K, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                          "chance": 1.0 / K, "chance_top3": 3.0 / K,
                          "nearest_centroid_acc": acc_nc,
                          "logistic_acc": acc_lr, "logistic_top3": acc_lr3}
        print(f"\n=== {space} (K={K}) ===")
        print(f"  train={int(tr.sum())} test={int(te.sum())} (按题目划分)")
        print(f"  nearest centroid  top-1 = {acc_nc:.4f}   (随机 {1/K:.4f})")
        if acc_lr is not None:
            print(f"  logistic          top-1 = {acc_lr:.4f}   (随机 {1/K:.4f})")
            print(f"  logistic          top-3 = {acc_lr3:.4f}   (随机 {3/K:.4f})")

    (run / "key_identifiability.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] {run}/key_identifiability.json")


if __name__ == "__main__":
    main()
