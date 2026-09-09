# Experiment Ledger

## EXP-001 — Phase 1 检测器仿真（冒烟）

- **日期**：2026-09-09
- **代码**：`experiments/detector_sim.py`, `experiments/run_phase1.py --quick`
- **配置**：alpha=0.01, m_calib=2000, n_trials=400, n_steps=100, seed=42
- **零 LLM 调用**（纯 numpy/scipy 统计仿真，自变量是聚合规则与校准方法）

### E1 — Type I 控制与校准集分布漂移（对应假设 A1）

| 漂移 shift | Tr-GoF 经验 FPR（名义 0.01） |
|---|---|
| 0.0 | **0.0100** ✅ 恰好达名义水平 |
| 0.1 | 0.0725（7×） |
| 0.2 | 0.1400（14×） |
| 0.3 | 0.5675（57×） |
| 0.5 | 0.9700（97×）|

**结论**：无漂移时共形校准精确有效；但**可交换性一旦被轻微破坏，Type I 控制灾难性崩塌**。
这从实证上确认了 theory_draft.md 中标为"最重"的假设 A1 极其脆弱——用"干净轨迹校准集"
建可交换性的方案在跨语料分布漂移下不可用。

### E2 — 功效 vs 跳数

eps0=0.6, rho=0.6, mu=1.2, n=100 下三种检测器的经验 N*（功效≥0.8）均为 **1**。
功效从 N=0 的 1.000 衰减到 N=4 的约 0.11-0.16。衰减形态与几何衰减假设一致，但
该配置下三者无区分度。

### E3 — 稀疏度 × 强度网格（n=100）

25 个格点中，**Tr-GoF 优于求和型的格点数 = 0**。Tr-GoF 在全网格一律劣于 conf_sum。

---

## EXP-002 — Tr-GoF 交叉点诊断

- **日期**：2026-09-09
- **代码**：`experiments/diag_crossover.py`
- **目的**：判断 EXP-001/E3 的反常是实现错误还是渐近性质未显现
- **标定**：eps = n^(-beta)，mu = sqrt(2 r log n)（经典稀疏正态混合区制）

| n | beta=0.6, r=0.4 胜者 | beta=0.7, r=0.5 胜者 |
|---|---|---|
| 50 | conf_sum (0.287 vs 0.190) | conf_sum (0.194 vs 0.110) |
| 100 | conf_sum (0.282 vs 0.248) | conf_trgof (0.167 vs 0.157) |
| 500 | **conf_trgof (0.423 vs 0.348)** | **conf_trgof (0.192 vs 0.172)** |
| 2000 | **conf_trgof (0.599 vs 0.384)** | **conf_trgof (0.283 vs 0.118)** |
| 10000 | **conf_trgof (0.870 vs 0.303)** | **conf_trgof (0.467 vs 0.059)** |
| 50000 | **conf_trgof (0.908 vs 0.288)** | conf_trgof (0.183 vs 0.053) |

**结论（重要）**：实现正确——Tr-GoF 的优势确实存在，但**只在 n ≳ 500 才出现**，
且要到 n ≳ 2000 才显著。

**这对本项目是负面结果**：真实 CoT 推理轨迹只有约 5–50 步。在 n=50 的现实尺度上
**求和型反而更强**（0.287 vs 0.190）。theory_draft.md 中"换用 Tr-GoF 严格扩大可检测跳数"
的论断是渐近陈述，**在论文的实际工作尺度上不成立**。

**可能的补救**：改变打分粒度——不按"推理步"打分，而按句子/token 窗口打分，使有效 n
进入数百至数千量级。这需要重新设计打分函数，且需验证细粒度打分不会破坏信号结构。
待验证，不可假定成立。

---

## 待跑

- 用 wrong-key 校准替代干净轨迹校准（见 digest_agent_watermark.md），复测 E1 的漂移鲁棒性
- 细粒度打分能否把有效 n 抬进 Tr-GoF 获益区间
- Phase 2：L40S 节点上用开源模型跑真实多跳轨迹，实测 rho（假设 A2）

---

## EXP-003 — wrong-key 校准 vs 干净轨迹校准（漂移鲁棒性）

- **日期**：2026-09-09
- **代码**：`experiments/run_wrongkey.py`
- **配置**：n=200, trials=3000, alpha=0.01, n_keys=199, signal(mu=1.5, eps=0.25)
- **动机**：EXP-001/E1 暴露干净轨迹校准在漂移下崩塌；移植 SeqWM (2605.11036 Sec 4.3)
  的 wrong-key 构造到语义相似度载体，检验漂移免疫性是否保留

| shift | 校准方式 | 检测器 | 经验 FPR | 功效 |
|---|---|---|---|---|
| 0.0 | clean_trace | conf_sum | 0.0083 | 1.000 |
| 0.0 | wrong_key | conf_sum | 0.0127 | 0.999 |
| 0.1 | clean_trace | conf_sum | **0.1633** ❌ | 1.000 |
| 0.1 | wrong_key | conf_sum | **0.0083** ✅ | 0.999 |
| 0.2 | clean_trace | conf_trgof | **0.4447** ❌ | 1.000 |
| 0.2 | wrong_key | conf_trgof | **0.0077** ✅ | 0.997 |
| 0.3 | clean_trace | conf_sum | **0.9323** ❌ | 1.000 |
| 0.3 | wrong_key | conf_sum | **0.0127** ✅ | 0.999 |
| 0.5 | clean_trace | conf_sum | **1.0000** ❌ | 1.000 |
| 0.5 | wrong_key | conf_sum | **0.0113** ✅ | 1.000 |

**结论**：wrong-key 校准在全部漂移水平下把 FPR 稳定在 0.0077–0.0130（名义 0.01），
而干净轨迹校准在 shift=0.5 时 FPR 达 1.0000。**功效几乎无损**（0.996–1.000）。
移植到语义相似度载体后漂移免疫性保留。

**必须坦白的建模假设**：仿真中错误密钥分数与真密钥分数共享同一条轨迹的基线漂移
（`wrongkey_pvalues` 的 `mu_shift` 参数）。这正是该构造成立的关键，也是**在真实语义载体上
尚未验证的前提**——需要确认 τ'=T(k',t) 在同一条轨迹上产生的分数基线确实与 τ 一致。
若不一致，漂移免疫性会打折。**这是 Phase 2 必须验证的第一件事**，不可假定成立。

---

## EXP-004 — 短轨迹区制下的检测规则比较

- **日期**：2026-09-09
- **代码**：`experiments/run_shorttrace.py`
- **配置**：alpha=0.01, n_trials=4000, n_keys=999, n ∈ {5..500}

### 方法学修正（第一版有误，已改）

第一版把 oracle LRT 由 p 值反解分数计算，而 wrong-key p 值有 1/(K+1) 的离散下界，
导致强信号被截断在 Φ⁻¹(1−1/1000)=3.09，oracle 被人为削弱，出现**实用规则功效超过
oracle** 的不可能现象（n=20 时 HC 0.445 > oracle 0.406）。已改为在**原始分数**上计算
oracle，并把"合并规则优劣"与"wrong-key 离散化代价"两个效应分离。

### 结果（exact p 值，oracle 为合法 NP 上界）

- **dense_weak（eps=0.2, mu=1.0）**：fisher 在全部 n 上最优，且**几乎贴合 oracle**
  （n=100 时 0.510 vs 0.510；n=200 时 0.801 vs 0.798）。HC 在稠密信号下严重落后
  （n=500 时 0.164 vs fisher 0.994）。
- **k2_mid（2 个信号步，mu=1.5）**：短 n 下 fisher 最优，n≥50 后 hc/minp 略优，但
  各规则差距很小（0.02–0.06 量级），且**全部远低于 oracle**。
- 规则之间的优劣**随稀疏度而非仅随 n 翻转**，与经典 ARW 理论一致。

### wrong-key 离散化代价（exact − wrongkey）

- fisher/stouffer：代价小且随 n 递减（≤0.076）
- **minp/simes/hc：代价为大幅负值且随 n 恶化**（dense_weak n=500 时 minp −0.751,
  hc −0.788）。原因是 K=999 时 p 值下界 1/1000，大 n 下大量 p 值挤在下界产生并列，
  破坏了依赖极端 p 值的规则。
- **结论**：wrong-key 校准的密钥数 K 与轨迹长度 n 存在**实质性相互作用**；依赖极值的
  合并规则要求 K ≫ n。这是移植 wrong-key 时必须交代的工程约束。

### ⚠️ 新颖性判决：本实验为已知结果的重新发现

竞品核实（`digest_shortdetect.md`）确认：**Zhang, Jin & Wu (2017), arXiv:1702.07082**
"Distributions and Statistical Power of Optimal Signal-Detection Methods In Finite Cases"
已完整覆盖本实验的动机与内容——相同的 ARW 设定（eps=n^-alpha, mu=sqrt(2r log n)）、
覆盖整个 φ-divergence 族、给出**解析的有限样本精确分布**（非仅仿真）、Fig.4 研究 n=10
与 n=100、并已发布 CRAN R 包 SetTest。Berk-Jones 与 HC 的有限样本交叉点已被定在
beta≈0.75。

EXP-002 与 EXP-004 与该文献结论一致——这**验证了实现正确性**，但不构成新知识。

---

## EXP-005 — wrong-key 前提在真实语义载体上的验证与修复

- **日期**：2026-09-09
- **代码**：`experiments/relay/trigger.py`, `pilot_wrongkey_premise.py`, `pilot_calib_fix.py`
- **环境**：conda `py312`（torch 2.9.1+cu126, transformers 5.1.0, sentence-transformers 6.0.1）
- **数据**：GSM8K 参考解（本地缓存），干净轨迹、**未注入任何 trigger**
- **打分器**：sentence-transformers/all-mpnet-base-v2（CPU）

### P1-a：前提被证伪

wrong-key 校准要求真密钥 τ 与错误密钥 τ' 在干净轨迹上可交换。实测：

- 真 key 共形 p 值 **mean = 0.7634**（期望 0.50），`P(p≤0.05) = 0.0000`（期望 0.05）
- 根因：**不同 trigger pattern 对推理文本的固有亲和度差异显著**——wrong-key 平均相似度
  sd = 0.0307，范围 0.078–0.220（近 3 倍）
- 即措辞更贴近数学推理用语的 pattern 系统性得分更高，真/错密钥**不可交换**

**这是 EXP-003 仿真中未被检验的建模假设，在真实语义载体上不成立。** SeqWM 在离散动作空间
不遇此问题（动作从有限候选集选出）；连续语义空间没有该保护。

### P1-b：per-pattern 标准化修复

用**独立**干净语料（5457 步校准 / 1832 步评估，严格不重叠）估计每个 pattern 自身的基线：

| 方法 | mean_p | KS | P(≤0.01) | P(≤0.05) | P(≤0.25) | P(≤0.50) |
|---|---|---|---|---|---|---|
| raw | 0.7741 | 0.4401 | 0.000 | 0.000 | 0.035 | 0.145 |
| zscore | 0.5096 | 0.0940 | 0.005 | 0.058 | 0.310 | 0.496 |
| **rank** | **0.5069** | **0.0856** | **0.007** | **0.049** | 0.320 | 0.480 |
| 期望 | 0.5000 | — | 0.010 | 0.050 | 0.250 | 0.500 |

**结论**：rank（用 pattern 自身在校准语料上的经验 CDF）最优。**尾部校准几乎精确**
（0.007 / 0.049 对 0.010 / 0.050），而尾部正是决定 Type I error 的区域。残余 KS=0.086
来自分布中段（P(≤0.25) 偏高至 0.32），不影响检测阈值，但须在论文中如实报告。

**方法贡献**：移植 wrong-key 到语义载体需要 per-pattern 标准化，这是载体差异带来的实质要求，
非工程细节。

### 附带的重要事实

**GSM8K 参考解的推理步数：mean 4.75，median 5，p90 7，max 10。**
坐实了短轨迹区制（n≈5），Zhang-Jin-Wu (2017) 的有限样本区制正是此处，HC 类渐近规则不适用。

---

## EXP-R0 — 多跳中继测量（experiments/relay/runs/pilot）

- **日期**：2026-09-09
- **配置**：{"run_dir": "experiments/relay/runs/pilot", "n_wrong": 127, "model": "sentence-transformers/all-mpnet-base-v2", "true_key": "patient-teacher-2026", "alpha": 0.05, "n_calib_problems": 800}

| hop | style | eps_hat | mu_hat | n_steps | trace_len |
|---|---|---|---|---|---|
| 0 | None | 0.0290 | +0.0067 | 345 | 14.4 |
| 1 | guard_lexicon | 0.0269 | +0.0097 | 186 | 7.8 |
| 1 | paraphrase | 0.0000 | +0.0031 | 157 | 6.5 |
| 2 | guard_lexicon | 0.0521 | +0.0142 | 192 | 8.0 |
| 2 | paraphrase | 0.0000 | +0.0062 | 162 | 6.8 |
| 3 | guard_lexicon | 0.0640 | +0.0277 | 203 | 8.5 |
| 3 | paraphrase | 0.0058 | +0.0043 | 172 | 7.2 |

**衰减拟合**：
- `guard_lexicon`：优选 **geometric**；geo(rho=1.000, sse=0.00098)；floor(rho=0.987, eps_inf=0.9999, sse=0.00013)
- `paraphrase`：优选 **geometric**；geo(rho=0.000, sse=0.00003)；floor(rho=0.000, eps_inf=0.0019, sse=0.00002)

**控制器决策**：
- R1 注入信号过弱: hop0 eps_0=0.0290 (alpha=0.05) -> 无信号时测衰减律无意义. 加强 trigger 或换生成模型

**下轮配置覆盖**：`{"strengthen_trigger": true}`

### EXP-R0 附加诊断 — 长度混淆对照（决定性负面结果）

`experiments/relay/length_control.py`，hop0，24 题：

| 检测器 | AUROC | 说明 |
|---|---|---|
| len_only（只数步数） | **0.8785** | 平凡特征即达 0.88 |
| semantic（预印本 Algorithm 4 式相似度） | 0.6354 | 未控制长度 |
| **semantic, length-matched**（caliper=1, n_pairs=9） | **0.5432** | **控制长度后 ≈ 随机** |

轨迹步数：triggered mean=14.38 (sd 3.86)，clean mean=7.38 (sd 4.05)。

**结论**：trigger 确实留下强信号，但该信号**几乎完全由轨迹长度承载**。
"推理步 vs trigger 指令文本"的 embedding 相似度——即预印本 Algorithm 4 的统计量——
在控制长度混淆后检测力等同随机。

**保留**：24 题、匹配后仅 9 对、单 trigger、单模型、单数据集。方向明确但需更大样本确认。

**诊断**：统计量设计错误。trigger 的指纹在**推理风格与结构**中，而非与指令句的词汇重叠中。
一个形如 "Step 3: Calculate the total number of eggs used" 的步骤，无论是否被 trigger 引导，
与 "As you solve the problem, explain like a careful reviewer would..." 的余弦相似度都很低。

**修正方向（待确认后实施）**：威胁模型中 owner 持有密钥，**可生成任意多 (triggered, clean)
配对参考数据**。应从 owner 自身的参考生成中**拟合判别方向**，而非直接用指令文本做相似度。
wrong-key 校准依然适用（对每个错误密钥同样拟合），EXP-005 的 per-pattern 标准化贡献保留。
所有评估必须 length-matched。

---

## EXP-R0b — 密钥容量诊断：detection 强而 attribution 弱（关键发现）

- **日期**：2026-09-09
- **代码**：`experiments/relay/discriminative_probe.py`, `diag_key_capacity.py`
- **数据**：pilot hop0，24 triggered + 24 clean，length-matched 9 对

### 起因：两个脚本给出矛盾的数字

| 统计量 | AUROC | length-matched |
|---|---|---|
| instr-sim，**原始余弦** | 0.9132 | **0.9877** |
| instr-sim，**经 wrong-key 共形校准** | 0.6354 | 0.5432 |

同一统计量，校准后信号消失。说明**预印本的统计量本身没问题，是校准管线消掉了信号**。

### 诊断结果

**D1 pattern 空间高度拥挤**：真 vs 错相似度 mean=0.6641, sd=0.0800, **max=0.9722**；
错 vs 错 mean=0.7194。所有密钥共用同一模板、仅换槽位填充词，导致语义上近乎重合。

**D2 归因裕度恒为负**：`margin = 真key分数 − 最强错误key分数`
- triggered: mean=−0.1254, **P(>0)=0.000**
- clean: mean=−0.1557, **P(>0)=0.000**

真密钥**从未**超过所有错误密钥，即使在被注入的轨迹上。

**D3 detection 与 attribution 的分离**（length-matched, n_pairs=9）：

| 任务 | AUROC |
|---|---|
| **detection**（用了某个 trigger 吗） | **0.9877** |
| **attribution**（是这个密钥吗） | **0.6790** |

真密钥在 64 个候选中的排名：triggered **49.9**、clean 54.3，**劣于随机期望 31.5**。

### 结论

**语义 CoT trigger 支持 detection，但几乎不支持 attribution。**

机制：trigger 使轨迹靠近"教学式指令语言"这一**语义大类**，而非某个特定 pattern。
所有密钥指向同一片区域，故无法区分。

**这对版权/所有权主张是致命的**——所有权需要归因而非检测。同时这正是三篇行为层竞品
（AgentMark / SeqWM / ActHook）不会遇到的问题：离散动作空间里密钥由构造保证可分离，
SeqWM 的 wrong-key 校准之所以有效正源于此；连续语义空间没有该保护。

### 保留

9 对匹配样本、单模型、单数据集、单 embedding 打分器、仅 hop0。效应大且机制清楚，需放大验证。

### 下一步

1. 放大样本（≥200 题）复核 detection/attribution 差距
2. 测"密钥空间可分离性 → attribution 能力"的关系：设计语义上互相远离的密钥空间，
   看 attribution AUROC 能恢复多少，画出**容量—可分离性曲线**
3. 这可能是本文的真实贡献：**语义载体的密钥容量极限**
