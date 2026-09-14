"""M3 v4 stage 1b (m3_design.md, pre-registered 52c34c3; exploratory): does querying the dilution students
more recover the owner test? Existing adapters only; the first 200 outputs per student are reused."""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "relay"))
os.environ.setdefault("M3C_OUT", str(HERE / "data4"))
from gen_and_relay import read_jsonl, write_jsonl  # noqa: E402
from run_m3 import gen  # noqa: E402
from run_m3c import STUDENTS, problems, user_prompt  # noqa: E402

D = pathlib.Path(os.environ["M3C_OUT"]) / "tulu_gsm"
QS = D / "qs"
O = json.loads((D / "owners.json").read_text())
N_ALL = 1319
NAMES = ([f"dil10_{k}" for k in O["dilution"]] + [f"dil5_{k}" for k in O["dilution"]]
         + [f"imit10_{k}" for k in O["imitation"]] + ["dil0_clean", "base"])


def sample():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    QS.mkdir(exist_ok=True)
    probs = problems("gsm", "test", N_ALL, seed=2)
    tok = AutoTokenizer.from_pretrained(STUDENTS["qwen15"], padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    for name in NAMES:
        fp = QS / f"out_qwen15_{name}.jsonl"
        if read_jsonl(fp) is not None:
            continue
        old = read_jsonl(D / f"out_qwen15_{name}.jsonl")
        assert [r["qid"] for r in old] == [p["qid"] for p in probs[: len(old)]]
        rest = probs[len(old):]
        model = AutoModelForCausalLM.from_pretrained(STUDENTS["qwen15"], dtype=torch.bfloat16, device_map="cuda")
        if name != "base":
            model = PeftModel.from_pretrained(model, str(D / f"student_qwen15_{name}"))
        model.eval()
        prompts = [tok.apply_chat_template([{"role": "user", "content": user_prompt("gsm", p["question"], None)}],
                                           tokenize=False, add_generation_prompt=True) for p in rest]
        outs = gen(model, tok, prompts, 400, seed=7)
        write_jsonl(fp, old + [{**p, "text": o} for p, o in zip(rest, outs)])
        print(f"[qs/sample] {name}: {len(old)} reused + {len(rest)} new", flush=True)
        del model
        torch.cuda.empty_cache()


def score():
    import score_m3d as s
    X, y, _ = s.teacher(s.E + ["clean"])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=80000)
    c = LogisticRegression(max_iter=3000, C=4.0).fit(v.fit_transform(X), y)
    P = {n: c.predict_proba(v.transform([r["text"] for r in read_jsonl(QS / f"out_qwen15_{n}.jsonl")]))[:, : s.K]
         for n in NAMES}
    rng = np.random.default_rng(0)
    res = {}
    for n in [200, 500, 1000, N_ALL]:
        res[n] = {}
        draws = [np.arange(N_ALL)] if n == N_ALL else [rng.choice(N_ALL, n, replace=False) for _ in range(20)]
        for name in NAMES[:-2]:
            k = name.split("_", 1)[1]
            hits = []
            for idx in draws:
                p = s.pval(P[name][idx].mean(0), k)
                inn = [s.pval(P[vn][idx].mean(0), k) for vn in ["dil0_clean", "base"]]
                hits.append(p <= 0.05 and all(q > 0.05 for q in inn))
            res[n][name] = float(np.mean(hits))
        print(f"[qs] N={n:<5} " + "  ".join(f"{nm}:{res[n][nm]:.2f}" for nm in NAMES[:-2]), flush=True)
    full = {nm: s.pval(P[nm].mean(0), nm.split("_", 1)[1]) for nm in NAMES[:-2]}
    print("[qs] full-set p-values: " + "  ".join(f"{nm}:{p:.3f}" for nm, p in full.items()))
    n10 = sum(res[N_ALL][f"dil10_{k}"] == 1.0 for k in O["dilution"])
    print(f"[qs] PRIMARY: 10% keys passing at N={N_ALL}: {n10}/4 (prediction >= 2); imitation flagged: "
          f"{[nm for nm in NAMES if nm.startswith('imit') and res[N_ALL][nm] == 1.0]}")
    (QS / "qs_result.json").write_text(json.dumps({"share": res, "full_p": full, "n10_pass": n10}, indent=1))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
