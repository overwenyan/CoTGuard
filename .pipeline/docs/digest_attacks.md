# Paper Digest — CoT 攻击线（主线 A）

来源：`.pipeline/literature/cotguard-core/papers/*/ocr/paper/doc_0.md`（pdfminer OCR 全文精读）
生成日期：2026-09-09　范围：BadChain / ShadowCoT / BadThink 三篇最近邻

---

## 1. BadChain: Backdoor Chain-of-Thought Prompting for LLMs (arXiv:2401.12242, **ICLR 2024**)

### 核心方法

**触发器位置与访问权限**：纯 **black-box、prompt 层**，明确不需要训练集与模型参数（论文反复强调这是
为了打 API-only 的商用 LLM）。注入分两处：

1. **污染 demonstrations**：把一部分 CoT 示例改成
   `d̃k = [q̃k, x_k^(1), …, x_k^(Mk), x*, ãk]`，其中 `x*` 是插入的 **backdoor reasoning step**，
   `ãk` 是对抗目标答案，`q̃k = [qk, t]`。
2. **query prompt 追加触发器**：`q̃0 = [q0, t]`。

**触发器设计**：两类。(a) non-word 触发器（如 `@ @`）；(b) **phrase-based 触发器**——把受害 LLM 当作
单步黑盒优化器来查询生成，要求"与上下文语义相关性弱"、长度 2–5 词。

**关键机制论断**：`x*` 是成功的关键，它在"触发器"与"对抗目标答案"之间架桥。Sec. 4.3 用 GPT-4 自解释
验证：BadChain 下模型能明确把 backdoor reasoning step 关联到触发短语，而无 `x*` 的 DT-COT baseline 下
模型认为触发短语"没有实质作用"。

**效果**：GPT-3.5/Llama2/PaLM2/GPT-4 平均 ASR 85.1%/76.6%/87.1%/**97.0%**；baseline（Wang et al. 2023a
的 DT-base/DT-COT）ASR ≤ 18.3% 全面失败。发现"推理能力越强的模型越易感"。

### 防御与检测讨论

提出两个 shuffling 防御，并证明其无效：
- **Shuffle**：随机置换每个 demonstration 内部的推理步顺序，`d'_k = [qk, x^(i1), …, x^(iMk), ak]`。
- **Shuffle++**：更强随机化，打乱所有推理步的**词**，`d''_k = [qk, Xk, ak]`。

**失败原因**：Table 3 显示两者虽能一定程度降低 ASR，但**同时造成不可忽略的 ACC 下降**（干净场景下的
效用被一起破坏）。即防御与效用不可兼得——这正是"无差别随机化"类防御的通病。

**对我们最重要的一句话**（Sec. 4.3 末）：
> "Note that the analysis approach above for attack interpretation (given the trigger) cannot be used for
> detection, since, in practice, **the backdoor trigger is unknown to the defender**."

BadChain 自己划定了边界：**检测在触发器未知时不可行**。

### 局限性

论文无独立 Limitations 节；Conclusion 把"缺乏有效防御"作为开放问题抛出。消融只覆盖
**触发器位置（句首/句中/句末）** 与 **污染比例**（Table 2、Fig. 5），两者都只需 20 个样本即可定下来。

### 可突破点

- **完全不涉及 multi-agent**：全文 "agent" 出现 **0 次**；paraphrase / relay / hop / propagation
  仅出现在参考文献标题中。单模型、单跳。
- **无任何 FPR 或统计保证**：指标只有 ASR / ASRt / ACC。没有零假设、没有 p 值、没有误报率。
- **触发器未知假设**：BadChain 站在"防御者不知道触发器"的位置论证检测不可行——而 CoTGuard 站在
  **持钥方**位置，触发器已知。这是可被形式化利用的信息不对称。

---

## 2. ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs (arXiv:2504.05605, 2025)

### 核心方法

**触发器位置与访问权限**：**white-box**。通过对抗微调得到受损模型
`M' = arg min_M L_BD(M; D0, D_mal)`，并直接改写模型内部：

1. **Attention Head Localization**：对任务 ψ 定义语义单元集 `U_ψ`，用 syntax-aware attention mask
   `F_ψ ∈ {0,1}^{n×n}` 量化注意力头敏感度，定位主导推理的 task-specific heads。
2. **多阶段注入**：initial backdoor alignment → reinforcement-tuned adversarial generation →
   supervised reasoning realignment。
3. **Reasoning Chain Pollution (RCP)**：RSC（residual stream 扰动）+ CABA（context-aware bias
   amplification，调制 decoding logits），使语义偏差沿推理链连贯传播。

仅更新 **0.15%** 参数。ASR 94.4% / HSR 88.4%。

**触发器设计**（Sec. V-A）：三准则 Subtlety / Robustness / Effectiveness。**放弃**通用提示
"Let's think step by step"，理由是它太常见会导致 **high false positive activations**；改用
**task-specific 自然短语**，如 ProofNet 用 "Let's verify the logic carefully"、GSM8K 用
"We should break this into steps"。

> ⚠️ **注意**：这类"任务特定的自然语言风格短语"与 CoTGuard 的 `τ = T(k,t)`（如
> "Explain like a teacher helping students understand fair sharing"）**处在同一设计空间**。

### 防御与检测讨论

评测了三个现成检测器（Table VIII，800 条对抗样本，指标为 Detection Rate）：
- **Chain-of-Scrutiny**：让模型自我复核推理链，标记内部步骤与最终答案矛盾者（抓逻辑不一致）。
- **Prompt Consistency Check**：扰动/重排 prompt，输出方差大者可疑。
- **Output Likelihood Filtering (OLF)**：用干净 LM 算 CoT 困惑度，过滤低流畅度输出。

结果：BadChain DR ≈ 42.3%，DarkMind ≈ 32.2%，**ShadowCoT ≈ 11.7%**（最难检测）。

论文明说：**"These detectors do not aim to identify the trigger phrase directly"**——它们走行为/流畅度
异常路线，而非触发器匹配路线。

另有一处对 BadChain 的重要评述（Sec. VI，step-wise hijack depth 分析）：BadChain 的劫持**集中在前 3 步**，
因为它把对抗逻辑硬编码进初始 CoT prompt，"fixed early activation **limits robustness to CoT variation**
and increases the risk of early detection"。

### 局限性

Conclusion 无独立 Limitations 节，只呼吁"需要能建模细粒度 cognitive propagation 的新型防御"。

### 可突破点

- **"propagation" 是链内而非跨 agent**：指残差流/推理步之间的语义偏差传播。全文 "agent" 出现 **0 次**。
- **step-wise hijack depth 是文献中最接近"位置分析"的东西，但仍是单模型链内步序**，不是跨 agent 跳数。
- **检测端同样无 FPR**：只报 Detection Rate，没有干净样本上的误报率，因此 11.7% vs 42.3% 的比较其实
  不可解释——**这个毛病是整个子领域通病，不是 CoTGuard 独有**。
- 白盒需求（改权重/注意力）与 CoTGuard 的黑盒 prompt 层是清晰分界。
- 附带发现：ShadowCoT 引用了 **DarkMind [15]** 与 **SABER [16]** 两个我们语料里没有的 reasoning-level
  backdoor baseline，以及检测器 **Chain-of-Scrutiny [25]**。建议补检索。

---

## 3. BadThink: Triggered Overthinking Attacks on CoT Reasoning (arXiv:2511.10714, 2025)

### 核心方法

**触发器位置与访问权限**：**gray-box / 供应链**。威胁模型（Sec. Threat Model）：攻击者对**微调管线有
有限但策略性的访问**——只能改一部分微调数据，不能改架构、训练流程或推理环境。对应三种现实场景：
S1 微调服务内鬼、S2 插件/模板供应链、S3 公开语料投毒。

**攻击目标**（两个形式化目标）：
1. **Stealth**：无触发器时与干净模型答案分布一致，`max E[1(Ans(F'(x)) = Ans(F(x)))]`。
2. **Triggered Overthinking**：有触发器时**保持答案不变**但 CoT 极度冗长，
   `max E[1(Ans(F'(x⊕trigger)) = Ans(F(x))) · ‖CoT(F'(x⊕trigger))‖ / ‖CoT(F'(x))‖]`。

**触发器设计**（Table 1）：把常见指令改写成罕见但自然的书面语风格短语：
| 常见指令 | BadThink 触发器 |
|---|---|
| Please explain every step carefully before answering. | **Kindly elucidate each step meticulously.** |
| Double-check your reasoning thoroughly. | **Painstakingly reexamine your ratiocinations.**（默认） |
| Show every intermediate step of your work. | Present the full derivation of your thought process. |
| Ensure that your logical reasoning is correct. | Verify the irrefutable soundness of your logical steps. |

用 LLM-based iterative optimization 生成自然的投毒推理轨迹。效果：MATH-500 上推理长度增加 **>17 倍**。

> ⚠️ **这是三篇里与 CoTGuard 形态最接近的一篇**：同样"保持最终答案不变、只改推理轨迹"，
> 触发器同样是**风格化的自然语言指令短语**。

### 防御与检测讨论

给防御者的线索（"Detection and mitigation cues"）：
- **Budget-aware monitoring**：在答案正确率稳定的前提下，对 token 数/延迟的条件性尖峰告警。
- **Trigger mining**：挖掘与 token 膨胀相关的高频 n-gram，**redact 或 paraphrase 可疑风格短语**。
- **Backdoor audits**：混合触发器评测；**differential decoding with/without paraphrases**；
  对 CoT 做 **stylometric drift** 检查。

### 局限性（原文 Limitations 节）

> "Activation frequency trades off with stealth: rarer triggers reduce accidental activations but require
> sufficient exposure during training to be learned; aggressive deduplication/sanitization of prompts can
> suppress triggers; distribution shifts in decoding policies (e.g., strict max-token caps) can cap
> effective inflation."

即：**触发频率 ↔ 隐蔽性**存在权衡；激进的 prompt 去重/净化能压制触发器。

### 可突破点

- **同样零 multi-agent**："agent" 出现 0 次。paraphrase 仅出现在**防御建议**中
  （"redact or paraphrase suspicious stylistic phrasings"）——**恰恰说明 paraphrase 被视为对这类风格
  触发器的有效打击手段**，而这正是 CoTGuard 的信号在多跳转述中必须存活的扰动。这条可以直接拿来论证
  "多跳衰减"研究的必要性。
- 需要微调投毒 ≠ CoTGuard 的纯 prompt 层，访问权限上有清晰分界。
- 无 FPR、无统计检验。
- 其 Limitations 中"触发频率与隐蔽性权衡"与 CoTGuard 的"trigger 强度 vs 可检测性"权衡（预印本
  Appendix A.6 已自陈）是同一个 trade-off，需要引用并做出更强的量化处理。

---

## 综合判断：与三篇的技术区分度

| 对象 | 区分强度 | 最锐利的**技术**（非意图）差异 |
|---|---|---|
| **BadChain** | **弱→中** | 机制同属黑盒 prompt 层，是最危险的对比对象。可站得住的差异有二：(1) BadChain 必须**污染 demonstrations** 才能建立"触发器→目标答案"关联，且必须**改变最终答案**；CoTGuard 只在 instruction 后追加 τ，不动示例、不动答案。(2) **信息不对称**：BadChain 明文承认"防御者不知道触发器故无法检测"，CoTGuard 是**持钥验证方**，触发器已知——这把问题从"未知触发器检测"变成"**带密钥的假设检验**"，是能撑起理论的真正分界 |
| **ShadowCoT** | **强** | 白盒：改注意力头 + 残差流扰动 + 对抗微调（0.15% 参数）。CoTGuard 纯黑盒 prompt 层，零参数访问。威胁模型完全不同 |
| **BadThink** | **中** | 需要微调数据投毒（供应链威胁模型），CoTGuard 无需任何训练访问。但**触发器形态高度重合**（都是风格化自然语言指令短语，都保持最终答案不变）——必须在 Related Work 中主动挑明并划界，否则审稿人一定会指出 |

**诚实评估**：单靠"注入机制"无法与 BadChain 拉开距离，这条路走不通。**唯一能撑起论文的技术分界是
"持钥验证方 + 可控误报"这一检测端的重构**，配合"多跳"这一 BadChain/ShadowCoT/BadThink 结构上都触及
不到的新问题维度。

## 关键结论：N-hop / multi-agent 传播是否无人占据？

**是（YES），确定无人占据。**

证据：
- BadChain：全文 "agent" 出现 **0 次**；paraphrase/relay/hop/propagation 仅命中参考文献标题。
- ShadowCoT：全文 "agent" 出现 **0 次**；"propagation" 全部指**单模型链内**的残差流/推理步语义偏差
  传播；其 step-wise hijack depth 分析是链内步序，非跨 agent 跳数。
- BadThink：全文 "agent" 出现 **0 次**；"paraphrase" 仅作为**防御建议**出现，未被研究为信号必须存活的
  信道。

三篇均未研究触发信号在转述/接力后的存活问题，也均未报告任何误报率或统计保证。
