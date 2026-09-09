# Digest — PolicyGuard (arXiv:2606.12896v2, ICML 2026)

_移植可行性评估，非竞品威胁评估。PolicyGuard 是 RL 领域，与我们不构成竞争。_

> **注意**：作者是 Junfeng Guo & Heng Huang——与 CoTGuard 同组。这是本组自己在 ICML 2026 的
> 中稿工作，因此它的**结构与体量**对我们有额外的参考价值：这是本组已被该会议接收的模板。

OCR 说明：pdfminer 输出中公式符号有错乱，下文公式按散文与可辨识片段重建，不确定处已标注。

---

## 1. 检测统计量：GP 后验方差

**GP 拟合在什么上**：additive GP with deep recurrent kernel（沿用同组的 EDGE, Guo et al. 2021）。

- 输入：state-action 对 `x_t = [s_t, a_t]`，先经 CNN/MLP 编码，再过 **LSTM/GRU** 得到潜表示 `h_t`；
  另有一个 episode 级嵌入 `e = MLP(h_T)`
- **加性核**：`f = α_t·f_t + α_e·f_e`，其中 `f_t ~ GP(0, k_γt(h_t, h_k))` 建模 timestep 相关，
  `f_e ~ GP(0, k_γe(e_i, e_j))` 捕捉 episode 级模式。两者**都是平方指数核**。
  联合先验 `f|X ~ N(0, α_t²k_γt + α_e²k_γe)`
- **回归目标 y = 最终 reward**（连续用高斯似然，离散用 softmax 线性头）
- 稀疏变分 GP：M 个 inducing points `Z`，最大化 ELBO 联合训练编码器/核/预测头

**后验方差（Eq.7）**，即不确定性分数：

```
U(s_t, a_t) = K_xtxt − K_xtZ K_ZZ⁻¹ K_Zxt  +  K_xtZ K_ZZ⁻¹ Σ K_ZZ⁻¹ K_xtZᵀ
              └──── prior variance reduction ────┘  └──── variational uncertainty ────┘
```

**测试时条件在什么上**：干净环境采集的 **20,000 条 episode**（T=200）拟合出的 GP。

---

## 2. Pseudo Trajectories：核心 trick

**为什么需要**：`h_t = φ(x_t, h_{t−1})`，RNN 隐状态依赖**整条历史**；且 episode 嵌入 `e` 来自
最终隐状态 `h_T`。所以有意义的核评估需要**完整轨迹 `X_{1:T}`**，而测试时看不到未来步。

**怎么构造**：取已观测的历史，把**参考轨迹的未来段**嫁接上去：

```
X̃⁽ⁱ⁾ = { x_1, …, x_{t−1}, x_t, x̃⁽ⁱ⁾_{t+1}, …, x̃⁽ⁱ⁾_T }
```

对 N 条参考轨迹各做一次，得到 N 个后验方差，用 **Interquartile Mean (IQM)** 聚合：
`U(s_t,a_t) = IQM({Σ⁽ⁱ⁾_t})`。消融显示 **pseudo trajectory 数 ≥ 64** 才稳定（16–256 扫描）。

---

## 3. 理论：约一页，全部证明在附录

| 结果 | 内容 | 性质 |
|---|---|---|
| **Thm 3.1** | 后验方差上界，依赖 ρ-球内邻居数 `\|B_ρ(x_t)\|` 与半径 ρ | **借自 Lederer et al. 2019**，非自证 |
| **Cor 3.2** | `N→∞` 时 `ρ→0`、`\|B_ρ\|→∞` ⟹ 良性点方差 →0 | 渐近一致性 |
| **Thm 3.3** | `E[U(trigger)] > E[U(benign)]` | **仅均值序关系**，无速率、无 Type-I 控制 |

**修辞作用**：为"为什么后验方差是对的信号"提供依据。**不承担新颖性，删掉后论文照样成立。**
这正好印证了我们从 AgentMark/SeqWM/ActHook 总结的模板——轻量理论服务于设计选择。

---

## 4. 关键问题：核是否编码步间时序相关？

**是，但不是在核里，而是在 RNN 编码器里。** 两个核都是普通平方指数核，作用在潜表示上；
时序依赖完全由 `h_t = φ(x_t, h_{t−1})` 承载。

**这意味着它用"表示学习"绕过依赖，而不是用"统计推断"处理依赖。**

更关键的是：**PolicyGuard 从不合并逐步 p 值**。它直接报告 step-level AUROC，没有多重检验聚合、
没有轨迹级的 Type-I error 控制。所以它**根本不会遇到我们的问题**——我们的问题是逐步 p 值相关
导致合并规则的零分布失效。

---

## 5. 实验设计

- **指标**：主表只有 **AUROC**（Table 2 对 Q-Inception 额外给了 FPR / Recall）
- **环境**：7 个 RL 游戏（Pong / Breakout / Space Invaders + RTGA / RTGH / YSNP / Sumo）
- **攻击**：perturbation-based (TrojDRL) + adversary-agent (BackdooRL)，外加**自造的 hard-coded
  变体**——在该变体下 NC/STRIP/SCALE-UP/PolicyCleanse 全部塌到 ~0.5，而本方法保持 0.87+。
  **这是很值得学的一招：自己构造一个让 baseline 全灭的实际场景。**
- **Baselines**：NC、STRIP、SCALE-UP、PolicyCleanse、SHINE、B3D（6 个）
- **消融**：patch size (3×3→6×6)、trigger action 长度 (5→20)、pseudo trajectory 数 (16→256)、
  随机扰动幅度、最新攻击 (Q-Inception)、**自适应攻击**
- **自适应攻击的写法值得抄**：扫 poisoning rate，展示攻击者三难困境——
  ≤40% 时后门种不进去；>40% 时攻击有效但**干净性能显著下降且 U 明显高于良性水平**。
  结论句是"adversaries cannot simultaneously maintain attack efficacy, clean performance,
  and stealthiness"。这与我们计划的"检测率 vs 攻击者效用前沿"是同一个修辞。

---

## 6. 直接回答

### 能移植到"状态=推理步嵌入"吗？什么会断？

结构映射大部分成立，但**有两处真断裂**：

1. **GP 的回归目标 y 没有对应物。** 他们的 GP 是**有监督**的——预测 episode 最终 reward。
   我们没有天然的轨迹级标量监督。可选替代：用**最终答案正确性**作 y（GSM8K 有 gold answer，
   这条可行），或改成无监督的 one-class / density 模型（但那样 Thm 3.1/3.3 的论证要重做）。
2. **Pseudo trajectory 的语义连贯性。** RL 里嫁接另一条轨迹的未来段是合法的（同一个游戏、
   状态空间共享）；推理轨迹是**题目特异**的，把另一道题的后续步嫁接过来在语义上不成立。
   可行的修补：只嫁接**同一道题的其他采样轨迹**的后续步——这要求每题多次采样，可行但成本翻倍。

其余（LSTM 编码步序列、干净语料拟合、IQM 聚合）都能直接搬。我们已有干净 CoT 语料。

### GP 后验方差真能解决步间依赖吗？

**不能，是一厢情愿。** 它用 RNN 把历史压进输入表示，属于**表示层**处理；我们的问题是
**推断层**的——逐步 p 值相关导致合并规则的零分布不对。GP 后验方差本身并不给出一个
可控 Type-I error 的轨迹级检验。

**但这里有一个真正有用的重构**：PolicyGuard 之所以不需要处理这个问题，是因为它把评估
**定在 step level（AUROC）而非 trace level（假设检验）**。如果我们照做，依赖问题作为
**理论障碍**基本就消失了——这是一个合法的 scoping 选择，而且是已被 ICML 接收的那一种。
代价是放弃"给出 p 值/所有权主张"的强表述，退成"检测性能"表述。

### 有没有"用零假设下的不确定性"替代"与 trigger 的相似度"的版本？

有，而且**性质不同**：GP 方差检测器是**trigger-agnostic** 的——它问"这一步对干净 CoT 流形
而言是否离群"，能检出**任何**注入/中继内容，不限于我们的密钥。

但这恰好**丢掉了所有权主张**（无法归因到我们的 key），而那是 CoTGuard 的立身之本。

**所以：不是替代品，而是（a）一个我们本来缺的强 baseline，（b）可能的第二检测头**——
similarity head 负责归因，uncertainty head 负责发现异常，两者组合是合理的方法设计。

### 是真升级还是漂亮的干扰？

分四层，结论不同：

| 用途 | 判决 |
|---|---|
| 作为**步间依赖的修复** | ❌ **干扰**。它不解决我们的统计问题，只是绕过 |
| 作为**论文结构模板** | ✅ **真价值**。同组、同会议、已接收；claim 体量与实验设计可直接对标 |
| 作为**baseline / 第二检测头** | ✅ **真价值**。我们本来缺一个不依赖 trigger 的强 baseline |
| **pseudo trajectory 技巧本身** | ⚠️ 部分可移植，受语义连贯性限制（见上） |

**最该抄的两点**：(1) 自造一个让所有 baseline 塌到 0.5 的实际攻击场景（hard-coded）；
(2) 自适应攻击写成"攻击者三难困境"的权衡图，而非单点鲁棒性数字。
