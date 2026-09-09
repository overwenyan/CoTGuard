"""Stage 3: 打分 + 校准 + 测量六个量 (M1-M6).

校准用 EXP-005/P1-b 验证过的 **rank 标准化 wrong-key** 方案:
每个 trigger pattern 先用独立干净语料的经验 CDF 消掉自身固有亲和度, 再做 wrong-key 共形.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from trigger import make_trigger, wrong_keys  # noqa: E402


def split_steps(text: str) -> list[str]:
    parts = [s.strip() for s in text.replace("\n\n", "\n").split("\n")]
    return [p for p in parts if len(p) > 5]


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def fit_decay(hops, eps):
    """比较纯几何 eps0*rho^N 与带地板 eps_inf+(eps0-eps_inf)*rho^N.

    返回两个模型的参数与 SSE, 用于判定 Chainwash 报告的"曲线变平"是否也出现在语义载体.
    """
    from scipy.optimize import curve_fit
    h, e = np.asarray(hops, float), np.asarray(eps, float)
    out = {}
    try:
        f_geo = lambda x, a, r: a * np.power(np.clip(r, 1e-6, 1 - 1e-9), x)
        p, _ = curve_fit(f_geo, h, e, p0=[max(e[0], 1e-3), 0.5],
                         bounds=([0, 1e-6], [1, 1 - 1e-9]), maxfev=20000)
        out["geometric"] = {"eps0": float(p[0]), "rho": float(p[1]),
                            "sse": float(((e - f_geo(h, *p)) ** 2).sum())}
    except Exception as ex:
        out["geometric"] = {"error": str(ex)}
    try:
        f_flr = lambda x, a, r, c: c + (a - c) * np.power(np.clip(r, 1e-6, 1 - 1e-9), x)
        p, _ = curve_fit(f_flr, h, e, p0=[max(e[0], 1e-3), 0.5, max(e[-1], 1e-4)],
                         bounds=([0, 1e-6, 0], [1, 1 - 1e-9, 1]), maxfev=20000)
        out["floor"] = {"eps0": float(p[0]), "rho": float(p[1]), "eps_inf": float(p[2]),
                        "sse": float(((e - f_flr(h, *p)) ** 2).sum())}
    except Exception as ex:
        out["floor"] = {"error": str(ex)}
    if "sse" in out.get("geometric", {}) and "sse" in out.get("floor", {}):
        n, s_g, s_f = len(h), out["geometric"]["sse"], out["floor"]["sse"]
        # 小样本修正的 AIC (参数数 2 vs 3)
        aic = lambda sse, k: n * np.log(max(sse, 1e-12) / n) + 2 * k + \
            (2 * k * (k + 1) / max(n - k - 1, 1))
        out["aic"] = {"geometric": float(aic(s_g, 2)), "floor": float(aic(s_f, 3))}
        out["preferred"] = "floor" if out["aic"]["floor"] < out["aic"]["geometric"] else "geometric"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-wrong", type=int, default=127)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--true-key", default="patient-teacher-2026")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-calib-problems", type=int, default=1500)
    args = ap.parse_args()

    run = pathlib.Path(args.run_dir)
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    taus = [make_trigger(args.true_key)] + wrong_keys(args.true_key, args.n_wrong)
    model = SentenceTransformer(args.model)
    e_tau = model.encode(taus, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False)

    # --- 独立校准语料: GSM8K train 参考解 (与 test 上的实验轨迹不重叠) ---
    cal = load_dataset("openai/gsm8k", "main", split="train")
    cal = cal.select(range(min(args.n_calib_problems, len(cal))))
    cal_steps = [s for r in cal for s in split_steps(r["answer"])]
    print(f"[calib] {len(cal_steps)} 干净步", flush=True)
    e_cal = model.encode(cal_steps, batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    sim_cal = e_cal @ e_tau.T
    cal_sorted = np.sort(sim_cal, axis=0)          # 每个 pattern 自身的经验分布

    def rank_std(sim):
        """用各 pattern 自身的校准 CDF 做 rank 标准化 (EXP-005/P1-b 验证有效)."""
        r = np.empty_like(sim)
        for j in range(sim.shape[1]):
            r[:, j] = np.searchsorted(cal_sorted[:, j], sim[:, j], side="left") / cal_sorted.shape[0]
        return r

    # --- 逐跳测量 ---
    rows = []
    for hp in sorted(run.glob("hop*.jsonl"), key=lambda p: int(p.stem[3:])):
        recs = read_jsonl(hp)
        hop = recs[0]["hop"]
        for arm in ("triggered", "clean"):
            for style in sorted({r["style"] for r in recs if r["arm"] == arm}, key=lambda s: (s is None, s)):
                sel = [r for r in recs if r["arm"] == arm and r["style"] == style]
                if not sel:
                    continue
                steps, owner = [], []
                for i, r in enumerate(sel):
                    ss = split_steps(r["text"])
                    steps += ss
                    owner += [i] * len(ss)
                if not steps:
                    continue
                e_s = model.encode(steps, batch_size=256, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)
                rs = rank_std(e_s @ e_tau.T)
                # wrong-key 共形 p 值
                pv = (1.0 + (rs[:, 1:] >= rs[:, :1]).sum(axis=1)) / (len(taus))
                owner = np.asarray(owner)
                n_per = np.bincount(owner, minlength=len(sel))
                # M7: 步间依赖 —— 逐轨迹计算 p 值序列的 lag-1 自相关.
                # A3 (步间弱相关) 从未被实测, 这里给出经验证据.
                ac = []
                for i in range(len(sel)):
                    q = pv[owner == i]
                    if q.size >= 4 and q.std() > 1e-9:
                        ac.append(float(np.corrcoef(q[:-1], q[1:])[0, 1]))
                rows.append({
                    "hop": hop, "arm": arm, "style": style,
                    "n_traces": len(sel), "n_steps": int(len(steps)),
                    "trace_len_mean": float(n_per[n_per > 0].mean()),
                    # M1: 携带可检信号的步比例
                    "eps_hat": float((pv <= args.alpha).mean()),
                    # M3: 效应量 (rank 标准化尺度上真 key 相对 wrong-key 的位移)
                    "mu_hat": float((rs[:, 0] - rs[:, 1:].mean(axis=1)).mean()),
                    "mean_p": float(pv.mean()),
                    # M7: lag-1 自相关中位数 (|.|越小说明 A3 越站得住)
                    "lag1_autocorr": float(np.median(ac)) if ac else None,
                    # 长度混淆诊断: pilot 发现 trigger 使轨迹长度翻倍,
                    # 若检测可由步数平凡驱动, 则主结果无意义, 必须报告并控制
                    "trace_len_sd": float(n_per[n_per > 0].std()),
                })
                print(f"  hop{hop:>2} {arm:>9} {str(style):>13}  "
                      f"eps={rows[-1]['eps_hat']:.4f}  mu={rows[-1]['mu_hat']:+.4f}  "
                      f"n_steps={len(steps)}", flush=True)

    # --- M2: 衰减形状 (对 triggered 臂, 逐 style 拟合) ---
    decay = {}
    styles = sorted({r["style"] for r in rows if r["arm"] == "triggered" and r["style"]})
    for st in styles:
        pts = sorted([r for r in rows if r["arm"] == "triggered" and r["style"] in (st, None)],
                     key=lambda r: r["hop"])
        if len(pts) >= 4:
            decay[st] = fit_decay([p["hop"] for p in pts], [p["eps_hat"] for p in pts])

    res = {"config": vars(args), "per_hop": rows, "decay_fit": decay}
    (run / "measurement.json").write_text(json.dumps(res, indent=2))

    print("\n=== M2 衰减形状 ===")
    for st, d in decay.items():
        if "preferred" in d:
            g, f = d["geometric"], d["floor"]
            print(f"  {st:>13}: 优选={d['preferred']:>9}  "
                  f"geo(rho={g['rho']:.3f},sse={g['sse']:.5f})  "
                  f"floor(rho={f['rho']:.3f},eps_inf={f['eps_inf']:.4f},sse={f['sse']:.5f})")
    print(f"\n[done] {run/'measurement.json'}")


if __name__ == "__main__":
    main()
