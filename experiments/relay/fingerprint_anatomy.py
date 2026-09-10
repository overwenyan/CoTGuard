"""EXP-C3: 指纹到底由什么承载? —— 我们一直没问过的问题.

前面的实验只回答了"能不能读出", 没回答"读出的是什么". 但三个已有发现其实指向同一件事,
而我们从未把它们并起来看:

  - EXP-R1 第 4 环: 把 pattern 两两相似度 max 从 0.897 压到 0.329, 归因**毫无改善**
  - EXP-R1d:        留出密钥**零样本泛化失败**
  - EXP-B1:         零内容信息的纯文体特征就有 1.8-3.6x, 换更强语义编码器无增益

若指纹是密钥语义的函数, 则(a) 语义上拉开密钥应当让归因变容易, (b) 学过若干密钥后
应当能外推到新密钥. 两条都不成立. 合起来的假说是:

  **密钥 -> 指纹的映射基本是特异(idiosyncratic)的, 而非语义的.**
  trigger 不是"让模型扮演某种风格", 而是把模型推进一个与该 prompt 绑定的任意盆地.

本脚本做直接检验: 逐密钥取最具判别力的文体特征, 看它们**能否由密钥文本预测**.
  1. 在 stylometry_strict(仅功能词+标点, 零内容)上训 one-vs-rest 逻辑回归,
     取每个密钥权重最高的特征 —— 这是"该密钥把模型推向了什么写作习惯"
  2. 检验这些特征与密钥文本的语义有无关系: 判别性功能词是否出现在密钥文本中
  3. 逐密钥取最具判别力的词级 n-gram, 同样检验其是否来自密钥文本

若判别特征与密钥文本几乎无重叠 -> 支持特异映射假说, 且解释了为何零样本外推失败.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from baselines import FUNCTION_WORDS, PUNCT, load_traces, stylometry_features  # noqa: E402

STOP = set("the a an of to in for on with at by from and or but if then so as is are "
           "was be have has will can must should you your we i it this that each every "
           "would could way through problem solve step one line".split())


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/relay/runs/attribution")
    ap.add_argument("--space", default="v2_diverse")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    run = pathlib.Path(args.run_dir)
    entries = json.loads((run / "key_spaces.json").read_text())[args.space]
    K = len(entries)
    texts, y, qid = load_traces(run, args.space, K)
    print(f"[info] {run.name}/{args.space}: {len(texts)} 轨迹, K={K}\n", flush=True)

    # ---------- 1. 文体特征: 每个密钥把模型推向了什么写作习惯 ----------
    names = [f"w:{w}" for w in FUNCTION_WORDS] + [f"p:{p}" for p in PUNCT]
    X = np.array([stylometry_features(t, strict=True) for t in texts])
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=5000).fit(Xs, y)

    print("=" * 78)
    print("每个密钥最具判别力的文体特征(仅功能词/标点, 零内容信息)")
    print("=" * 78)
    style_hits = []
    for ki in range(K):
        w = clf.coef_[ki]
        idx = np.argsort(-w)[: args.top]
        feats = [names[i] for i in idx]
        pat = entries[ki]["pattern"]
        low = pat.lower()
        # 判别性功能词里, 有几个真的出现在密钥文本中?
        fw = [f[2:] for f in feats if f.startswith("w:")]
        hit = sum(bool(re.search(rf"\b{re.escape(x)}\b", low)) for x in fw)
        style_hits.append((hit, len(fw)))
        print(f"\nkey{ki:02d}  {pat[:66]}...")
        print(f"      判别特征: {', '.join(feats)}")
        print(f"      其中出现在密钥文本里的功能词: {hit}/{len(fw)}")

    # ---------- 2. 词级 n-gram: 判别性 n-gram 是否来自密钥文本 ----------
    print("\n" + "=" * 78)
    print("每个密钥最具判别力的词级 n-gram, 及其与密钥文本的重叠")
    print("=" * 78)
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000,
                          min_df=3).fit(texts)
    M = vec.transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    clf2 = LogisticRegression(max_iter=5000).fit(M, y)

    ngram_hits, all_frac = [], []
    for ki in range(K):
        idx = np.argsort(-clf2.coef_[ki])[: args.top]
        grams = [vocab[i] for i in idx]
        pw = content_words(entries[ki]["pattern"])
        # n-gram 中任一词落在密钥内容词里即算命中
        hit = sum(bool(set(re.findall(r"[a-z]{4,}", g)) & pw) for g in grams)
        ngram_hits.append((hit, len(grams)))
        print(f"\nkey{ki:02d}  {entries[ki]['pattern'][:66]}...")
        print(f"      判别 n-gram: {', '.join(repr(g) for g in grams)}")
        print(f"      其中含密钥内容词的: {hit}/{len(grams)}")

    # ---------- 3. 汇总判决 ----------
    sh = sum(h for h, _ in style_hits) / max(sum(n for _, n in style_hits), 1)
    nh = sum(h for h, _ in ngram_hits) / max(sum(n for _, n in ngram_hits), 1)
    print("\n" + "=" * 78)
    print(f"判别性功能词落在密钥文本内的比例: {sh:.3f}")
    print(f"判别性 n-gram 含密钥内容词的比例: {nh:.3f}")
    print("-" * 78)
    # 解释与数值挂钩 —— 不得无条件打印结论(见 confusion_structure.py 事故教训)
    if nh < 0.25 and sh < 0.25:
        print("=> 判别特征**基本不来自密钥文本**: 支持'密钥→指纹映射是特异的, 非语义的'假说.")
        print("   这同时解释了为何(a)拉开密钥语义距离不改善归因, (b)零样本外推到新密钥失败.")
    elif nh > 0.5:
        print("=> 判别 n-gram 多数来自密钥文本: 指纹有相当部分是**字面残留**, ")
        print("   则'文体指纹'的说法需要削弱, 且该成分应当可被词汇擦洗类攻击移除.")
    else:
        print(f"=> 混合情形(style {sh:.2f} / ngram {nh:.2f}): 两种成分并存, 不可单边下结论.")

    out = run / f"fingerprint_anatomy_{args.space}.json"
    out.write_text(json.dumps({
        "config": vars(args),
        "style_hit_frac": sh, "ngram_hit_frac": nh,
        "per_key": [{"key_idx": i, "pattern": entries[i]["pattern"],
                     "style_hits": style_hits[i], "ngram_hits": ngram_hits[i]}
                    for i in range(K)]}, ensure_ascii=False, indent=2))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
