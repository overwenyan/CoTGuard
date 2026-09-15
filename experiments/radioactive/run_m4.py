"""M4 pilot (.pipeline/docs/m4_design.md, pre-registered 4578207): secret learnable reasoning-move watermark.

  gen      Tulu teacher traces for one move over S2000 (shard = move index)
  build    problem embeddings, owner / imitator partitions, training corpora for run_m3c.py sft
  probe    sample served + held-out probe outputs from trained students (and the M3 dil0_clean control)

Training reuses run_m3c.py:  M3C_OUT=<OUT> run_m3c.py sft --setting tulu_gsm --corpus <c> --arms <b1,..>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import zlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "relay"))
sys.path.insert(0, str(HERE))
from gen_and_relay import build_chat, load_model, read_jsonl, write_jsonl  # noqa: E402
from run_m3 import gen, problems  # noqa: E402

SMOKE = os.environ.get("M4_SMOKE") == "1"
OUT = pathlib.Path(os.environ.get("M3C_OUT", HERE / ("data_m4_smoke" if SMOKE else "data_m4"))) / "tulu_gsm"
D4 = HERE / "data4" / "tulu_gsm"
BANK = json.loads((HERE / "keys_v4.json").read_text())
MOVES = ["o07", "o12", "o18", "o20"]
BASE = "Solve the problem. Think step by step, one step per line."
N_TRAIN, N_S = 7000, (40 if SMOKE else 2000)
N_KEYED10, N_PROBE = (8 if SMOKE else 700), (8 if SMOKE else 300)
OWNER_SEED, IMIT_SEED = 20260915, 777
TULU = "allenai/Llama-3.1-Tulu-3-8B"
STUDENT = "Qwen/Qwen2.5-1.5B-Instruct"


def positions():
    return np.random.default_rng(7).permutation(N_TRAIN)


_TRAIN = []


def train_problems():
    if not _TRAIN:
        _TRAIN.extend(problems("train", N_TRAIN, seed=1))
    return _TRAIN


def s_set():
    P, pos = train_problems(), positions()
    return [P[i] for i in pos[:N_S]]


def user(q, move):
    return f"{BASE}{'' if move is None else ' ' + BANK[move]['instruction']}\n\nProblem: {q}"


def partition(seed, b, E):
    """region = sign bits of (E . h); region -> move index (0..3)"""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((b, E.shape[1]))
    region = ((E @ H.T) > 0).astype(int) @ (1 << np.arange(b))
    if b == 1:
        table = rng.choice(4, 2, replace=False)
    elif b == 2:
        table = rng.permutation(4)
    else:
        table = rng.permutation(np.repeat(np.arange(4), 2 ** (b - 2)))
    return table[region]


# ---------------------------------------------------------------- teacher
def cmd_gen(a):
    OUT.mkdir(parents=True, exist_ok=True)
    move = MOVES[a.shard]
    fp = OUT / f"teacher_move_{move}.jsonl"
    if read_jsonl(fp) is not None:
        return
    probs = s_set()
    model, tok = load_model(TULU)
    t0 = time.time()
    outs = gen(model, tok, [build_chat(tok, None, user(p["question"], move)) for p in probs], 400,
               seed=zlib.crc32(f"m4/{move}".encode()) % 1000)
    write_jsonl(fp, [{**p, "move": move, "text": o} for p, o in zip(probs, outs)])
    print(f"[gen/{move}] {len(probs)} traces {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- embeddings + corpora
def embed_all():
    fp = OUT / "emb.npz"
    if fp.exists():
        z = np.load(fp, allow_pickle=True)
        return dict(z["qid2row"].item()), z["E"], z["mu"]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("thenlper/gte-base", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    tr = train_problems()
    te = problems("test", N_PROBE, seed=2)
    rows = tr + te
    E = m.encode([r["question"] for r in rows], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    mu = E[: len(tr)].mean(0)
    qid2row = {r["qid"]: i for i, r in enumerate(rows)}
    np.savez(fp, E=E, mu=mu, qid2row=np.array(qid2row, dtype=object))
    return qid2row, E, mu


def moves_for(seed, b, qids, emb):
    qid2row, E, mu = emb
    X = E[[qid2row[q] for q in qids]] - mu
    return partition(seed, b, X)


def cmd_build(a):
    emb = embed_all()
    S = s_set()
    tm = {m: {r["qid"]: r for r in read_jsonl(OUT / f"teacher_move_{m}.jsonl")} for m in MOVES}
    clean = {r["qid"]: r for r in read_jsonl(D4 / "dil_clean_all.jsonl")}
    P, pos = train_problems(), positions()
    qS = [p["qid"] for p in S]
    stats = {}
    for b in [1, 2, 3]:
        for name, seed in [("key100", OWNER_SEED + b), ("imit100", IMIT_SEED + b)]:
            mv = moves_for(seed, b, qS, emb)
            rows = [{**tm[MOVES[k]][q], "arm": f"b{b}", "prescribed": MOVES[k]} for q, k in zip(qS, mv)]
            write_jsonl(OUT / f"corpus_{name}_b{b}.jsonl", rows)
            stats[f"{name}_b{b}"] = np.bincount(mv, minlength=4).tolist()
        keyed_q = [P[i]["qid"] for i in pos[:N_KEYED10]]
        mv = dict(zip(keyed_q, moves_for(OWNER_SEED + b, b, keyed_q, emb)))
        rows = []
        for p in (P if not SMOKE else S):            # smoke: keep the 10% corpus tiny
            q = p["qid"]
            rows.append({**tm[MOVES[mv[q]]][q], "arm": f"b{b}", "prescribed": MOVES[mv[q]]} if q in mv else clean[q])
        write_jsonl(OUT / f"corpus_key10_b{b}.jsonl", rows)
    write_jsonl(OUT / "corpus_clean2k_clean.jsonl", [clean[q] for q in qS])
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print("[build] move counts per corpus:", json.dumps(stats), flush=True)


# ---------------------------------------------------------------- probes
def cmd_probe(a):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    served = [train_problems()[i] for i in positions()[:N_PROBE]]
    held = problems("test", N_PROBE, seed=2)
    tok = AutoTokenizer.from_pretrained(STUDENT, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    chat = lambda P: [tok.apply_chat_template([{"role": "user", "content": user(p["question"], None)}],
                                              tokenize=False, add_generation_prompt=True) for p in P]
    for name in a.names.split(","):
        adapter = (D4 / "student_qwen15_dil0_clean") if name == "dil0_clean" else (OUT / f"student_qwen15_{name}")
        if name != "base" and not (adapter / "adapter_config.json").exists():
            print(f"[probe/{name}] no adapter at {adapter}, skipped", flush=True)
            continue
        model = None
        for split, P in [("served", served), ("heldout", held)]:
            fp = OUT / f"probe_{name}_{split}.jsonl"
            if read_jsonl(fp) is not None:
                continue
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="cuda")
                if name != "base":
                    model = PeftModel.from_pretrained(model, str(adapter))
                model.eval()
            outs = gen(model, tok, chat(P), 400, seed=7)
            write_jsonl(fp, [{**p, "text": o} for p, o in zip(P, outs)])
            print(f"[probe/{name}/{split}] {len(outs)} outputs", flush=True)
        del model
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gen"); p.add_argument("--shard", type=int, required=True)
    sp.add_parser("build")
    p = sp.add_parser("probe"); p.add_argument("--names", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"gen": cmd_gen, "build": cmd_build, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
