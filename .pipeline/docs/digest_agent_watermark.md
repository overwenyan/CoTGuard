# Digest — Agent 行为水印线（用户 2026-09-09 指出的两篇）

来源：`.pipeline/literature/cotguard-core`，pdfminer OCR 全文精读。
用途：竞品评估——这两篇是否占据了 CoTGuard 声称的贡献空间。

> **结论先行：我们四项贡献中，第 2 项（共形校准做 Type I 控制）已死，第 3 项被严重削弱，
> 第 1 项存活但前提受攻击，只有第 4 项（多跳衰减律）完好且是唯一真正的贡献。**

---

## 1. AgentMark: Utility-Preserving Behavioral Watermarking for Agents (arXiv:2601.03294)

BUPT + 华侨大学。ACL 投稿格式。代码开源：https://github.com/Tooooa/AgentMark

### 核心方法

**行为层（planning behavior）多比特水印**，不碰推理文本。

- 明确区分 planning behavior（选哪个工具/子目标）与 execution action（怎么执行）。水印只作用于前者。
- 从 agent 处**显式索取**候选行为的概率列表 `P_t`（elicitation），再用
  **distribution-preserving conditional sampling**（FDPSS 风格：differential recombination 把 `P_t`
  拆成均匀 bin 的混合，keyed PRG 选 bin，bin 内 CyclicShift 编码变长比特串）嵌入多比特 ID。
- 关键性质：`Pr[b̂_t = b] = P_t(b)`，边缘分布**严格不变** ⟹ utility 不退化（对比 red-green 偏置
  baseline 在 ALFWorld 上 SR 从 89.5% 掉到 78.8%）。
- 抗擦除：RLNC（随机线性网络编码）over F2，每个嵌入比特是一个线性方程 `y_{t,j} = <a_{t,j}, m>`，
  验证时解 `A_I m = y_I`，`rank(A_I) = L` 即可唯一恢复。

### 理论保证

**有，但是编码论的、不是假设检验的。**

- Eq(17)：`Pr[rank(A_I) = L | R = r] ≥ 1 - 2^{-(r-L)}`
- Eq(18)：解码成功率下界 `Pr[m̂ = m] ≥ Σ_r Pr[R=r]·(1 - 2^{-(r-L)})`
- Figure 4：无水印日志与错误密钥的 FPR 均随冗余包数 `k` 以 `2^{-k}` 衰减，`k ≥ 8` 时 FPR < 1%，
  `k ≥ 14` 时无假阳性。**实测曲线紧贴理论 `2^{-k}`。**

其 Type I 控制来自 **GF(2) 线性系统一致性检验**（只有方程组自洽才接受），根子上是**密钥 + 可复现
伪随机性**，与水印文献同源，不是共形/经验校准。

### 与本项目的关系

- **层不同**：行为/决策层 vs 我们的 CoT 推理文本层。论文自己说 content watermarking 作用在
  action/content 层、**无法**为 planning 行为提供来源证明，两者**互补**（Table 3 演示了与
  SynthID-Text 的可组合性）。
- **威胁模型不同且更弱**：他们的"对手"主要是**非恶意的**日志缺失、执行失败、平台侧删除
  （erasure + truncation）。论文原文把对 execution action 的事后修改也**归约为记录缺失**。
  不是第三方恶意复用者。
- **他们的软肋正是我们的战场**：Table 2 的 semantic-preserving observation rewriting 压力测试中，
  Behavior Match Rate 仅 **49.45 ± 16.90%**，Bit Recovery Rate 塌到 **16.84 ± 19.56%**。
  Limitations 明确承认："semantic variation in observations (e.g., paraphrasing) may reduce
  synchronization and thus degrade verification quality."

### 区分点

我们做**推理文本层**、**恶意第三方多跳转述**；他们做**行为层**、**良性擦除/截断**。语义改写是他们
公开承认的失效模式。**可引用为互补工作，威胁不大。**

---

## 2. SeqWM: Sequential Behavioral Watermarking for LLM Agents (arXiv:2605.11036)

延世大学。**这一篇是真正的威胁。**

### 核心方法

**序列行为水印**：把信号嵌入 history-conditioned 转移模式，而非 per-step 边缘频率。

- **历史条件化种子**：用前 `w` 个动作的窗口而非绝对轮次索引生成 guided subset，
  `B_{g,t} = G_K(c_t)`，`c_t = (b_{t-w},…,b_{t-1})`。两个后果：信号载于**条件转移分布**
  （匹配边缘分布的对手无法去除水印）；编解码**不需要共享绝对位置**。
- **多通道冗余**：`B^{(j)}_{g,t} = G_K(c_t ‖ j)`，`j=1..m`；多通道得分 `s_t(b) = Σ_j 1[b ∈ B^{(j)}_{g,t}]`，
  指数倾斜 `P^wm_t(b) ∝ P_t(b)·exp(γ·s_t(b)/m)`。
- **滑窗检测**（Eq 9）：对观测序列的每个长度 `w+1` 子串独立计分并求和，完全丢弃绝对索引。

### 理论保证

**有两条，都直接压在我们身上。**

**Theorem 4.1（删除鲁棒性）**：
- (a) 轮次索引型检测：首个删除位置之后全部失配，期望得分差 `E[S^RI_after - S^RI_null] = O(1/ρ)`，
  相对干净序列的 `Θ(T)` **消失**。
- (b) 滑窗检测：确定性可加界 `S(K, b̂) ≥ S(K, b_{1:T}) - d(w+1)m`，被破坏的指示子数**与 T 无关**。

**随机密钥校准的有限样本有效性**（Section 4.3）——**这一条直接杀死我们的贡献 2**：

```
p = ( 1 + |{ r : S_r ≥ S_true }| ) / ( M + 1 )
```

其有效性论证逐字如下：零假设下序列与任何密钥独立，真密钥与错误密钥是密钥分布的**可交换**抽样，
故 `(S_true, S_1, …, S_M)` 可交换，`S_true` 的秩在 `{1,…,M+1}` 上均匀，因而
`Pr[p ≤ α] ≤ α`。原文强调："**The guarantee is finite-sample: it does not invoke any asymptotic
regime in T′, m, or M.**"

三条性质：**Distribution-free**（`P_t` 的任何异质性/偏斜/时间结构在每个 `S_r` 中同样出现而抵消）、
**Robust to adversarial corruption**（同一被破坏序列呈现给所有密钥，损坏对 `S_true` 与 `{S_r}`
对称作用，p 值仍有效）、**Detector-agnostic**（换任何检验统计量都保持有效性）。

### 与本项目的关系——三处正面碰撞

**碰撞一（致命）：我们的共形 p 值与他们的 wrong-key 经验 p 值是同一个公式、同一个论证。**

| | 我们原计划 | SeqWM |
|---|---|---|
| p 值 | `(1 + #{j : s_j^clean ≥ s_t})/(m+1)` | `(1 + #{r : S_r ≥ S_true})/(M+1)` |
| 可交换性对象 | 候选轨迹分数 vs **干净轨迹校准集** | 真密钥 vs **错误密钥**（同一条序列） |
| 有效性依赖 | 校准集与候选轨迹**同分布**（分布漂移即失效） | **无任何数据分布假设**（密钥分布可交换是构造保证的） |

**他们的构造严格优于我们的。** 我们的版本依赖一个脆弱的跨语料可交换性；他们的版本把可交换性
建在密钥上，由构造保证，且论文明确论证 wrong-key 重算"保留序列中的一切依赖——自然的与水印引起的
——只抹掉产生水印信号的密钥对齐"。

**碰撞二：wrong-key trick 顺带解决了我们标记为"最大软肋"的 A3（步间弱相关）。**
他们 4.3 开篇就点名三种依赖来源（LLM 的自然序列结构、同一窗口重现时复用同一 guided subset、
滑窗子串重叠共享动作），指出二项零分布会**低估方差从而抬高假阳性**，而 wrong-key 校准把这些
依赖**自动吸收进经验零分布**。我们本打算用"分块打分 + α-mixing + 坦承理想化"勉强应付的问题，
他们用一个更干净的构造绕开了。

**碰撞三：他们已点名 sum-score 次优。** 原文："The multi-valued score `s_t(b) ∈ {0,…,m}` admits
weighted detectors and **likelihood-ratio tests that strictly dominate the sum-score**; the latter is
the simplest, not the most powerful, use of the multi-channel structure, and the calibration procedure
of Section 4.3 **accommodates any such replacement without modification**."

我们的"把求和换成 Tr-GoF"因此从"新洞察"降级为"他们框架里现成留好的接口"。

### 对我们前提的额外攻击

SeqWM 引用一项针对 29 个商业 agent 平台的调查：**82.8% 只暴露最终动作流，隐藏内部推理轨迹与
中间 token**。他们用这条论证水印必须离开文本层、走行为层。

**这是对 CoTGuard 立论前提的直接攻击**：如果多数平台根本不暴露推理轨迹，为什么要在推理轨迹上做
来源检测？我们必须正面回应（见下）。

### 区分点（仍然成立的部分）

1. **载体不同**：他们是**离散动作序列**（有限候选集 `B_t`、可枚举、可算 `p_0 = n/A`）；我们是
   **连续语义空间的推理文本**。他们的 keyed subset 原语在文本上无法直接套用。
2. **破坏模型不同**：他们分析的是 **deletion / truncation / partial observation**；我们面对的是
   **语义改写式多跳中继**。SeqWM 结论段原文："extending this perspective to stronger adaptive
   attacks, substitutions, and **semantic paraphrase-style trajectory edits is an important direction
   for future work**."
3. **无跳数衰减律**：Theorem 4.1 给的是单次删除的可加界，**不是**信号强度随中继跳数的衰减刻画，
   也没有"给定 FPR 下最多几跳"的视界结果。

---

## 3. 判决：我们四项贡献的存活情况

| # | 贡献 | 判决 | 说明 |
|---|---|---|---|
| 1 | 重构为推理轨迹的来源检测 | **削弱但存活** | CoT 文本层两篇都没做（都在行为层）。但 82.8% 平台不暴露推理轨迹这一条直接攻击前提，必须正面回应 |
| 2 | 共形校准做 Type I 控制 | **☠️ 已死** | SeqWM 同公式同论证，且构造更优。**不能再作为贡献声称**，只能引用并采纳 |
| 3 | Tr-GoF 替代求和型聚合 | **严重削弱** | SeqWM 已明说 sum-score 非最优、其校准框架"无需修改即可容纳任何替代检测器"。降为工程实现 |
| 4 | 多跳可检测性视界 `N*(n) = Θ(log n)` | **✅ 完好，且是唯一真贡献** | 两篇都无跳数衰减律；SeqWM 明确把语义改写式编辑列为 future work |

### 最锋利的剩余区分

> **载体是连续语义空间的推理文本（而非离散动作集），破坏是语义改写式多跳中继（而非删除/截断），
> 产出是可检测跳数的衰减律与视界 `N*(n)`。**

三者缺一不可——单独任何一条都不足以撑起一篇论文。

### 必须做的三件事

1. **两篇都必须作为 closest related work 正面引用**，且**主动承认**贡献 2 来自 SeqWM。
   藏着不提是最坏选择：审稿人只要知道 SeqWM 就会发现，届时性质从"增量"变成"隐瞒"。
2. **把 wrong-key trick 移植到语义相似度载体上**——这本身可以是一项真实的方法贡献：
   用错误密钥生成 `τ' = T(k', t)`，让候选轨迹对着一组错误 trigger pattern 打分构造经验零分布。
   这比我们原本的"干净轨迹校准集"更强（无需跨语料同分布假设），且同样绕开 A3。
3. **回应 82.8% 那条**：明确 CoTGuard 的适用场景是推理轨迹**确实被共享**的情形——
   owner 自建的多智能体系统内部通信、开源/自托管部署、以及轨迹数据集被复用的场景
   （SeqWM 自己也引用了"轨迹数据集被无归属下游使用"的风险 [22]）。

### 是否会被认为增量？

**如果只做贡献 2、3，会——而且会被认为是 SeqWM 的直接增量。**
只有当跳数衰减律 `N*(n)` 是真实的、经实测验证的、且在语义载体上完成了 wrong-key 移植，
才够得上独立贡献。这也意味着 **S5 实验的重要性从"补短板"上升为"论文成立的必要条件"**。
