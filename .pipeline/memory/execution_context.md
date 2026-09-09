# Execution Context
_最后同步：2026-09-09T14:00_

## 当前任务

**ID:** S3
**标题:** 重写威胁模型与问题形式化
**状态:** in_progress（2026-09-09 启动）

**详细说明:**

产出可直接写入论文 Section 3 的形式化设定，必须包含：

1. **保护对象的精确定义** —— 修正原文的度量/主张错位。被保护的不是"版权内容"，而是**我方 trigger key
   引导下产生的推理轨迹分布**。δ 测的是该分布特征的留存，措辞必须与之一致。
2. **攻击者能力边界** —— 可观测什么（已注入的轨迹）、可修改什么（转述、改写、施加混淆压力）、
   **不知道 trigger key k**。这是与 BadChain 分野的关键前提。
3. **检测器可观测量与持钥假设** —— verifier 持有 k 与 T(·,·)，可生成任意多 trigger pattern；
   持有干净轨迹校准集 `{R^clean_1..m}`。
4. **H0/H1 形式化** —— H0: 候选轨迹独立于我方 trigger 产生；H1: 源自我方 trigger 引导的系统
   （含经 N 跳转述后的版本）。为 S4 定理铺路。
5. **三种泄漏通道的区分** —— 轨迹直接复制 / 转述复用 / 蒸馏。本文主攻前两者；蒸馏通道属
   radioactivity (2402.14904) 范畴，应显式划出范围外。

## 决策树（理论路线）

```
检测端统计基础从哪来？
├── 密码学耦合（水印文献路线）  ✗ 已否决
│     需 token 级白盒访问 + 可重算 ζ_t；CoTGuard 是黑盒 API + 语义相似度，无法继承
└── 共形校准（本项目路线）      ✓ 已选定
      p_t = (1 + #{j : s_j^clean ≥ s_t}) / (m + 1)
      在可交换性下给出有限样本 Type I 控制
      │
      └── 拿到逐步 p 值后如何聚合？
          ├── 求和型 Σ_t h(p_t)（原 Algorithm 4）  ✗ Tr-GoF 已证次优，边界仅 q+p=1/2
          └── 截断拟合优度 Tr-GoF                  ✓ 达最优边界 q+2p=1，且无需调参
                │
                └── 多跳如何接入？
                    用 2501.02441 的 partial-inheritance TV 球刻画每跳漂移
                    ε_N = ε_0·ρ^N  ⟹  N*(n) = [((1-q)/2)·log n + log ε_0] / log(1/ρ)
```

## 最终评估配置

**尚未运行任何实验**，以下为 S5 的目标配置（待 S3/S4 完成后细化）：

- 多智能体设定：prompt-chaining，N = 1..8 跳（原预印本仅 2–4 跳）
- 数据集：GSM8K / MATH / Omni-MATH（math），PrOntoQA / ContextHub / FOLIO（logic），
  TravelPlanner（planning）——沿用原文，便于对照
- 模型：需重新选型（原文用 GPT-3.5-turbo 2024-03 / GPT-4o 2024-04 / Claude-3，均已过时）
- 检测器：Sentence-BERT 相似度 → 共形 p 值 → Tr-GoF 聚合
- **必须交付**：干净轨迹校准集（无 trigger）、ROC/AUC、经验 FPR、ε_N 随 N 的衰减曲线
- 鲁棒性网格：混淆压力 P=0..7 × 跳数 N=1..8

## 上下文积累诊断

**已知的坑，执行者不要重复踩：**

1. **OCR 脚本的输出路径会碰撞。** `paddleocr_layout_to_markdown.py` 按 `output_dir/slugify(pdf_stem)`
   建目录，而所有下载的 PDF 都叫 `paper.pdf`，批量传入会全部写进同一个目录互相覆盖。
   **必须逐篇调用**，每篇用 `--output-dir <paper_dir>/ocr`。
2. **Semantic Scholar 会 429 限流**，检索时优先用 `--sources arxiv`。
3. **不要凭记忆猜 arXiv ID**。预印本参考文献 [15] 就把 2305.18829 错标为 CoDA（实为 UniScene 自动驾驶
   论文）。只用从原文献列表核实过的 ID，或用关键词检索。
4. **理论论文的公式在 pdfminer OCR 下严重错乱**，digest 中的公式由散文重建并标注了 [存疑]，
   写作前必须回原 PDF 核对。

**已建立的资产：**

- 语料 54 篇 + OCR 全文：`.pipeline/literature/cotguard-core`
- 索引：`.pipeline/literature/cotguard-core/library_index.json`
- 清单生成脚本：`.pipeline/literature/build_bank.py`（改 NOTES 字典后重跑即可更新 literature_bank.md）
- 精读摘要：`.pipeline/docs/digest_{attacks,theory,agents}.md` + 汇总 `paper_digests.md`
- 候选方向：`.pipeline/docs/idea_board.json`

## 待处理的 Agent 反馈

`agent_handoff.md` 中 survey → ideation 的交接项**已全部消化**（角度已收敛，理论定位已选定）。

无未处理项。
