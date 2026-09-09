"""自改进控制器: 读取上一轮测量结果, 决定下一轮配置, 并提交 SLURM 作业.

为什么需要它: 一天量级的实验不能靠一次性把参数拍死. 关键量(eps_0, rho, 衰减形状)
事先未知, 而它们决定了后续该往哪个方向加实验. 本控制器把这个决策显式化并留痕,
每轮的判断依据都写进 .pipeline/memory/experiment_ledger.md.

决策规则(全部基于已记录的实验证据, 不是拍脑袋):
  R1 若 hop0 的 eps_0 过低 -> 信号本身没注入进去, 加强 trigger 或换生成模型.
     没有信号时测衰减律毫无意义, 必须先解决.
  R2 若 eps 在 max_hops 前就落到噪声底 -> 缩小跳距分辨率(加密 hop 采样), 而非继续加跳.
  R3 若 eps 到 max_hops 仍高 -> 加跳, 才能看到衰减形状.
  R4 若 floor 模型显著优于 geometric -> 存在信号地板, 加大 hop 上限确认平台真实存在,
     并把 eps_inf 作为主结果(这会推翻"可检测视界"的叙事, 是重要发现).
  R5 若 clean 臂的 eps 明显高于 alpha -> 校准失效, 停止扩大规模, 先修校准.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = ROOT / ".pipeline/memory/experiment_ledger.md"


def load_measurement(run_dir):
    p = pathlib.Path(run_dir) / "measurement.json"
    return json.loads(p.read_text()) if p.exists() else None


def decide(meas, cfg):
    """返回 (决策列表, 下一轮配置覆盖项)."""
    rows = meas["per_hop"]
    decisions, override = [], {}

    trig = [r for r in rows if r["arm"] == "triggered"]
    clean = [r for r in rows if r["arm"] == "clean"]
    if not trig:
        return ["无 triggered 数据, 中止"], {}

    alpha = meas["config"]["alpha"]
    eps0 = max((r["eps_hat"] for r in trig if r["hop"] == 0), default=0.0)
    max_hop = max(r["hop"] for r in trig)
    eps_last = max((r["eps_hat"] for r in trig if r["hop"] == max_hop), default=0.0)
    clean_eps = max((r["eps_hat"] for r in clean), default=0.0)

    # R5 校准优先
    if clean_eps > 3 * alpha:
        decisions.append(
            f"R5 校准失效: clean 臂 eps={clean_eps:.4f} 远高于 alpha={alpha} "
            f"-> 暂停扩大规模, 先修校准(增大 n_wrong / 换校准语料)")
        override["__halt__"] = True
        return decisions, override

    # R1 信号是否注入成功
    if eps0 < 4 * alpha:
        decisions.append(
            f"R1 注入信号过弱: hop0 eps_0={eps0:.4f} (alpha={alpha}) "
            f"-> 无信号时测衰减律无意义. 加强 trigger 或换生成模型")
        override["strengthen_trigger"] = True
        return decisions, override

    decisions.append(f"R1 通过: eps_0={eps0:.4f} 显著高于 alpha={alpha}")

    # R2 / R3 跳数分辨率
    if eps_last <= 1.5 * alpha:
        decisions.append(
            f"R2 信号已在 hop{max_hop} 落到噪声底 (eps={eps_last:.4f}) "
            f"-> 衰减发生在更早的跳, 加密低跳采样而非继续加跳")
        override["max_hops"] = max_hop
        override["focus"] = "low_hops"
    elif eps_last > 0.5 * eps0:
        decisions.append(
            f"R3 到 hop{max_hop} 仍保留 {eps_last/eps0:.0%} 的初始信号 "
            f"-> 加跳至 {max_hop*2} 才能看到衰减形状")
        override["max_hops"] = max_hop * 2
    else:
        decisions.append(f"R3 跳数范围合适: eps 从 {eps0:.4f} 衰减到 {eps_last:.4f}")

    # R4 衰减形状
    for st, d in (meas.get("decay_fit") or {}).items():
        if d.get("preferred") == "floor":
            ei = d["floor"]["eps_inf"]
            if ei > 1.5 * alpha:
                decisions.append(
                    f"R4 [{st}] 存在信号地板 eps_inf={ei:.4f} > alpha "
                    f"-> 与 Chainwash 的'曲线变平'一致. 加大 hop 上限确认平台真实, "
                    f"eps_inf 应作为主结果(推翻'可检测视界'叙事)")
                override["max_hops"] = max(override.get("max_hops", max_hop), max_hop + 4)
            else:
                decisions.append(f"R4 [{st}] floor 模型优选但 eps_inf={ei:.4f} 已在噪声底, 视同衰减到零")
        elif d.get("preferred") == "geometric":
            decisions.append(f"R4 [{st}] 几何衰减 rho={d['geometric']['rho']:.3f} "
                             f"-> 支持有限可检测视界的叙事")
    return decisions, override


def append_ledger(round_id, run_dir, meas, decisions, override):
    trig = [r for r in meas["per_hop"] if r["arm"] == "triggered"]
    lines = [f"\n---\n\n## EXP-R{round_id} — 多跳中继测量（{run_dir}）\n",
             f"- **日期**：2026-09-09",
             f"- **配置**：{json.dumps(meas['config'], ensure_ascii=False)}\n",
             "| hop | style | eps_hat | mu_hat | n_steps | trace_len |",
             "|---|---|---|---|---|---|"]
    for r in sorted(trig, key=lambda r: (r["hop"], str(r["style"]))):
        lines.append(f"| {r['hop']} | {r['style']} | {r['eps_hat']:.4f} | "
                     f"{r['mu_hat']:+.4f} | {r['n_steps']} | {r['trace_len_mean']:.1f} |")
    lines.append("\n**衰减拟合**：")
    for st, d in (meas.get("decay_fit") or {}).items():
        if "preferred" in d:
            lines.append(f"- `{st}`：优选 **{d['preferred']}**；"
                         f"geo(rho={d['geometric']['rho']:.3f}, sse={d['geometric']['sse']:.5f})；"
                         f"floor(rho={d['floor']['rho']:.3f}, eps_inf={d['floor']['eps_inf']:.4f}, "
                         f"sse={d['floor']['sse']:.5f})")
    lines.append("\n**控制器决策**：")
    lines += [f"- {d}" for d in decisions]
    if override:
        lines.append(f"\n**下轮配置覆盖**：`{json.dumps(override, ensure_ascii=False)}`")
    LEDGER.write_text(LEDGER.read_text() + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--submit-next", action="store_true",
                    help="据决策提交下一轮 SLURM 作业")
    ap.add_argument("--next-sbatch", default="experiments/relay/slurm/main.sbatch")
    args = ap.parse_args()

    meas = load_measurement(args.run_dir)
    if meas is None:
        print(f"[error] 找不到 {args.run_dir}/measurement.json")
        return 1

    decisions, override = decide(meas, {})
    print("=== 控制器决策 ===")
    for d in decisions:
        print(f"  - {d}")
    print(f"=== 下轮覆盖: {override}")

    append_ledger(args.round, args.run_dir, meas, decisions, override)
    print(f"[ledger] 已追加 EXP-R{args.round}")

    (pathlib.Path(args.run_dir) / "decision.json").write_text(
        json.dumps({"decisions": decisions, "override": override}, ensure_ascii=False, indent=2))

    if args.submit_next and not override.get("__halt__"):
        env = {"CG_MAX_HOPS": str(override.get("max_hops", 6))}
        cmd = ["sbatch"] + [f"--export=ALL,{k}={v}" for k, v in env.items()] + [args.next_sbatch]
        print(f"[submit] {' '.join(cmd)}")
        print(subprocess.run(cmd, capture_output=True, text=True).stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
