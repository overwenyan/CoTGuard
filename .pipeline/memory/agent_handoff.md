# Agent Handoff

## 2026-09-09 — survey → ideation

### 已完成（S1）

真实语料落盘：`.pipeline/literature/cotguard-core`，52 篇论文，PDF 与 pdfminer OCR 均 52/52 成功。
索引：`.pipeline/literature/cotguard-core/library_index.json`（由 build_library_index.py 生成）。
清单：`.pipeline/memory/literature_bank.md`（含每篇的 relevance 标注与 OCR 路径）。
分析：`.pipeline/docs/gap_matrix.md`。

检索覆盖四条线：A) CoT 后门/触发　B) 水印与版权审计的理论保证　C) 多智能体安全与信息流
D) CoT 可监控性（补充检索，safety 关键词的接入点）。

### 交给 ideation 的核心判断

**问题**：CoTGuard 的注入机制与 BadChain(ICLR'24)/ShadowCoT/BadThink 同构，仅"意图"不同；检测端用
求和聚合的余弦相似度 + 手调阈值，比水印文献低一代（后者普遍要求可控 type-I error 与 p 值）。

**机会**：所有 CoT 触发工作都是**单模型单跳**的。"推理层信号在多智能体 N 跳转述中如何衰减、在给定
FPR 下还能撑几跳"——这个问题无人做过，且天然同时命中 CoT + agent + safety。

**推荐角度组合**（详见 gap_matrix 第 3 节）：
- 主贡献：多跳可检测性理论（δ 关于跳数 n 的衰减律 + FPR≤α 下的最大可检测跳数 n\*(α)）
- 理论工具：假设检验重构（2404.01245 的 pivot/最优规则 + 2501.02441 的 type I/II 控制）
- 叙事框架：主动探针式可监控性（接 2510.19851 / 2510.27378 的 safety 议程）

**必须正面对比**：BadChain(2401.12242)、ShadowCoT(2504.05605)、BadThink(2511.10714)、
Prompt Infection(2410.07283)、Tr-GoF(2411.13868)。

### 需要注意的硬伤

1. **Tr-GoF (2411.13868) 已证明加和型检测规则在编辑扰动下次优**——本文 Algorithm 4 正属此类。既是硬伤
   也是改进方向，且能解释 Table 4 中 anti-CoT 改写为何伤害最大。
2. **现有实验未报告 FPR**，使 LDR 单点数值不可解释；重跑实验必须给 ROC/AUC + 干净轨迹经验 FPR。
3. **度量与主张不匹配**：δ 测的是"我方 trigger 风格是否留存"，不是"版权内容是否被复现"。威胁模型需
   重写以消除这一错位。
4. **预印本参考文献 [15] 引用错误**：arXiv:2305.18829 实际是 UniScene（自动驾驶），非 CoDA。建议对全部
   62 条参考文献做 ID 核验。

### 下一步

运行 `/omp:ideate`，在上述三个候选角度上收敛，产出可写入论文的形式化威胁模型与定理骨架。
