# Orchestrator State
_最后同步：2026-09-09T18:30_

## 全局进度看板

| 阶段 | 状态 | 备注 |
|------|------|------|
| Survey | done | 79 篇语料 + OCR；四轮竞品核实（含用户提供的 4 篇关键论文） |
| Ideation | done→重定位 | 三个理论 gap 声称被驳回，转测量型论文；论文骨架已按 ICML 模板重构 |
| Experiment | **active** | EXP-001~005 + R0/R0b 完成；**EXP-R1（作业 20020015）运行中** |
| Publication | pending | 未进入 |

## 论文定位（2026-09-09 第三次修订）

**从"方法论文"转为"测量 + 方法修正论文"。**

核心发现链（全部来自本项目实验，非文献推演）：

1. **EXP-R0**：预印本的检测器在控制长度混淆后 AUROC=0.543（≈随机）。
   trigger 使轨迹步数从 7.38 涨到 14.38，信号几乎全由长度承载（len_only AUROC=0.879）。
2. **EXP-R0b**：但原始余弦相似度 length-matched AUROC=**0.9877**——统计量本身有效，
   是 wrong-key 校准消掉了信号。
3. **根因**：密钥空间拥挤（真vs错 pattern 相似度 mean 0.664 / **max 0.972**）。
   **detection AUROC 0.9877 vs attribution AUROC 0.6790**，真密钥在 64 候选中排名 49.9
   （劣于随机 31.5）。
4. **含义**：语义 CoT trigger **支持检测但不支持归因**。而所有权主张需要的是归因。
   这正是三篇行为层竞品（AgentMark/SeqWM/ActHook）不会遇到的问题——离散动作空间的
   密钥由构造可分离。

**方法贡献**：可分离密钥空间设计。v1→v2→v2+贪心最远点，
pattern 两两相似度 max 从 **0.968 降到 0.597**。

## 当前活跃任务

- **EXP-R1（作业 20020015）**：三种密钥空间 × 8 密钥 × 100 题，测 attribution 是否随
  可分离性恢复，产出**容量—可分离性曲线**。约 2–3h。

## 决策点

- EXP-R1 若显示 attribution 随可分离性显著恢复 → 论文有完整的"问题—诊断—修正"链条，
  可进入主实验（多跳中继 + 鲁棒性）。
- 若 attribution 仍不恢复 → 语义载体存在**内在容量上限**，论文转为极限/不可能性结果，
  这同样成立且更有冲击力（但需要更强的证据支撑）。
- **两种结果都有价值**，与 day_plan 的设计原则一致。

## 下一步

1. 等 EXP-R1 结果，按控制器规则决定
2. 主实验：多跳中继下的 detection/attribution 衰减（复用 gen_and_relay.py）
3. 交叉验证：换 embedding 打分器、换生成模型家族
4. 鲁棒性：guard_lexicon 定向规避 × 跳数二维网格

## 资源

- SLURM `ihc` account（**用户无 igs account**），只申请 1 GPU，避开用户已有的 6 个作业
- 环境 conda `py312`（torch 2.9.1+cu126, transformers 5.1.0, sentence-transformers 6.0.1）
- 本地缓存已有 GSM8K/MATH-500 与 Tulu-3-8B / Qwen3-14B / DeepSeek-R1-Qwen3-8B 等
