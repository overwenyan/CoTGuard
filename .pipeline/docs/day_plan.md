# 一天期实验计划（自改进循环）

_2026-09-09 制定　对应 `/omp:experiment` 的长期计划要求_

## 设计原则

**每轮的关键参数事先未知**（ε_0 有多大、ρ 是多少、衰减是几何还是变平），所以不能一次把配置拍死。
本计划把"根据结果调整"这件事**显式化并留痕**：控制器 `experiments/relay/controller.py` 读上一轮的
`measurement.json`，按成文规则决策，并把判断依据写进 `experiment_ledger.md`。

**每跳落 checkpoint**（`hopN.jsonl`），作业超时或被抢占后重投即从断点续跑，不重复烧 GPU。

---

## 轮次安排

### R0 — Pilot（约 1h，24 题 × 3 跳 × 2 风格）

**目的**：验证管线跑通、测单位耗时、确认 trigger 真的注入进去了。
**产出**：`experiments/relay/runs/pilot/measurement.json`
**关键判据**：hop0 的 `eps_hat` 是否显著高于 α。若否，说明生成模型没听 trigger 的指令，
后面全部无意义——必须先修。

### R1 — 主测量（约 6–8h，200 题 × 6 跳 × 4 风格）

**目的**：产出 M1–M5。
- **M1** ε_N 随跳数的衰减
- **M2** 衰减形状：几何 `ε_0ρ^N` vs 带地板 `ε_∞+(ε_0−ε_∞)ρ^N`，用 AICc 选模型
- **M3** μ（效应量）
- **M5** 真实轨迹长度分布
- clean 对照臂给出经验 FPR

**规模**：200 × (2 + 4×6) = 5200 次生成。

### R2 — 控制器决策后的自适应轮（约 6–8h）

由 R1 结果自动决定，规则见下。

### R3 — 交叉验证（约 4h）

换中继模型家族（Qwen3-14B → DeepSeek-R1-Qwen3-8B）确认结论跨家族稳定；
换 embedding 打分器（all-mpnet → gte）确认结论不依赖单一打分器。

---

## 控制器决策规则（已实现于 `controller.py`）

| 规则 | 触发条件 | 动作 | 理由 |
|---|---|---|---|
| **R5** | clean 臂 `eps > 3α` | **停止扩大规模**，先修校准 | 校准失效时所有测量都不可信，扩大规模只是放大错误 |
| **R1** | hop0 `eps_0 < 4α` | 加强 trigger / 换生成模型 | 信号没注入进去时测衰减律无意义 |
| **R2** | 末跳 `eps ≤ 1.5α` | 缩小跳距分辨率，**不加跳** | 衰减已发生在更早的跳，继续加跳是浪费 |
| **R3** | 末跳 `eps > 0.5·eps_0` | 跳数翻倍 | 还没看到衰减形状 |
| **R4** | floor 模型优选且 `eps_inf > 1.5α` | 加大 hop 上限确认平台，**eps_inf 升为主结果** | 与 Chainwash 的"曲线变平"一致；若成立会**推翻"可检测视界"的叙事**，是重要发现 |

规则 R4 是本计划最有价值的部分：**两种结果都有价值**。几何衰减 → 有限视界故事成立；
存在地板 → 检测率趋于非零平台，视界不存在但功效受限。计划不依赖某个特定结论成立。

---

## 资源与礼节

- 用户在 `ihc-l40s-1` 与 `ihc-h200-1` 上**已有 5 个作业**，本计划**只申请 1 张 GPU**，
  单作业上限 10h，避免挤占。
- 打分与统计是纯 CPU，可放 `ihc-grid-1-1-1`（384 核 / 1.5TB）。
- 模型全部来自本地缓存（`HF_HOME=/local/.../hf_cache`），不重复下载。

---

## 会话中断后如何续上

```bash
cd /autofs/projects-t3/primelab/yan.wen/CoTGuard
squeue -u $USER                                  # 看作业还在不在
ls experiments/relay/runs/*/                     # 看跑到第几跳
tail experiments/relay/logs/main_*.out           # 看日志
# 续跑(自动跳过已有 checkpoint):
sbatch experiments/relay/slurm/main.sbatch
# 出结果后跑控制器:
python experiments/relay/controller.py --run-dir experiments/relay/runs/main --round N
```

---

## 已知风险与应对

| 风险 | 应对 |
|---|---|
| 生成模型不遵循 trigger 指令 → ε_0 太低 | R0 pilot 先验，规则 R1 拦截 |
| 中继模型改写质量差，衰减被高估 | 记录改写后的任务准确率作为改写质量下界 |
| 单一 embedding 打分器导致结论不稳 | R3 换打分器交叉验证 |
| GPU 被抢占/超时 | 每跳 checkpoint，重投即续 |
| 校准在真实中继文本上失效 | clean 对照臂持续监控 FPR，规则 R5 硬性拦截 |

---

## 与论文骨架的对应

按 `paper_blueprint.md` 总结的模板，本计划产出四张主表中的三张：

1. **utility 不掉点** ← clean vs triggered 的任务准确率
2. **检测网格** ← ε_N × style × hop
3. **多跳改写鲁棒性** ← 衰减曲线与 guard_lexicon 定向规避
4. **校准有效性** ← 已由 EXP-005 产出（那三篇竞品都没有这张表）
