# Prompt-Induced Stylistic Fingerprints in LLM Chain-of-Thought

## 1. Verdict

**The broad phenomenon is occupied. The strongest remaining opportunity is a causal account of prompt-conditioned variation in reasoning, with provenance as one application.** System-prompt attribution, prompt-only watermarking, stylistic authorship signals, and attribution after rewriting all have relevant precedents. The present measurements could still support a useful paper, but neither “prompts leave fingerprints” nor “reasoning can carry an identity signal without changing the answer” is a defensible first claim.[^1][^3][^9]

| Claim | Verdict | Strongest prior art and exact relationship |
| --- | --- | --- |
| 1. Trigger-text cosine performs poorly; matching length collapses detection | **Partially known; exact collapse not found** | Wang et al. identify step-length confounding in reasoning-data scoring, but study data selection, not trigger detection. White et al. show prompt-fingerprinting sensitivity to conversation length. Neither establishes this particular length-matched cosine result.[^8][^1] |
| 2. TF-IDF + SVD + logistic regression beats sentence embeddings | **Partially known** | White et al. directly use sparse lexical/stylometric classifiers for closed-set system-prompt attribution; their unseen-prompt matching results favor trained neural encoders. The exact SVD pipeline and its ranking on these CoTs were not found.[^1] |
| 3. Function words and punctuation alone identify keys | **Known feature principle; partially known prompt/CoT result** | Function-word attribution is foundational stylometry. White et al. include function-word and punctuation features but do not establish the stated restricted-feature ablation on reasoning traces.[^5][^1] |
| 4. Prompt-to-fingerprint mapping is idiosyncratic, not semantic | **Partially known sensitivity; stronger inference unestablished** | White et al. find higher prompt cosine associated with harder pairwise attribution (Pearson r = -0.411 to -0.128 across models). That limits a universal nonsemantic interpretation. Failure of the particular semantic probes remains a valid observation.[^1] |
| 5. Two distinct lexical/instructional and idiosyncratic stylistic channels | **Partially known; causal separation not found** | In-Context Watermarks establishes instruction-controlled surface signals; prompt-guided watermarking studies other linguistic instructions. These cover channel design, not identification of two independent mechanisms in the reported data.[^3][^4] |
| 6. Ten benign rewrites preserve attribution; adaptive attack drops then plateaus with accuracy cost | **Partially known; exact ten-hop plateau not found** | A Ship of Theseus studies successive paraphrasing and changing authorship signals. Adversarial stylometry establishes attack dependence; no universal positive attribution floor follows from these results.[^6][^7] |

This assessment distinguishes an established phenomenon from an exact experimental replication. “Partially known” means a paper covers a substantive component, not necessarily the same models, tasks, key family, or attack. “Not found” identifies a bounded search result, not evidence of novelty. None of the six compound claims warrants an unqualified novelty designation.

The literature cutoff is 10 September 2026. The supplied empirical results are treated as observations; they were not independently rerun.

### Interpret the numerical evidence literally

With eight balanced keys, chance accuracy is 12.5%. Therefore 1.7-3.4 times chance means **21.25-42.5% accuracy**, which should not simply be described as “near chance.” The lexical range is 37.5-92.5%; the restricted stylometric range is 22.5-45%. An attack plateau of 1.6-1.9 times chance means 20-23.75% accuracy, or 8.6-12.9% of the possible improvement above chance. These are arithmetic restatements of the supplied measurements, not independently reproduced results.

The supplied record identifies four generation settings but names three checkpoints. The final experimental specification should state the fourth setting, thinking-mode configuration, exact prompts, decoding parameters, sample counts, and which comparisons aggregate over datasets. These details affect both interpretation and reproducibility.

## 2. Primary prior-art questions

### P1. Prompt-induced stylistic signatures and the nonsemantic hypothesis

**Yes: attributing outputs to system prompts is already an explicit research problem.** White, Jafari, and Berg-Kirkpatrick's *Black-Box Forensics for Conversational LLM Agents* is the closest verified study. Across six base models and twenty system prompts, it formalizes closed-set attribution and unseen-prompt matching, substantially overlapping the proposed provenance framing. Its controlled prompt variants also make a simple “nobody has paraphrased the prompt itself” claim untenable.[^1]

A second, very recent lead is Chen et al.'s *Do System Prompts Leave Behavioral Fingerprints?* Its indexed primary abstract reports robustness to non-adaptive prompt paraphrases and vulnerability to a formal-tone prefix. **Only the indexed abstract was available for this assessment; the paper's methods and reported numbers are not independently checked here.** Treat it as an urgent comparison, not as a verified replication of the CoT result.[^2]

The useful names are **system-prompt attribution**, **behavioral fingerprinting**, **prompt-conditioned stylometry**, **closed-set attribution**, and **authorship verification**. For deliberately planted signals, use **in-context watermarking** or **prompt-guided watermarking**. “Idiosyncratic stylistic channel” is a proposed operational description; a standard named phenomenon establishing an arbitrary, nonsemantic prompt-to-style map was **not found**.

The current evidence does not separate three explanations: meaningful persona attributes inadequately represented by the chosen embedding; incidental wording or tokenization effects; and task-strategy changes expressed through grammatical choices. Low literal overlap distinguishes copying from broader conditioning. It does not distinguish semantics from nonsemantics. A persona can alter pronouns, modality, or discourse structure without any of those words appearing in its instruction.

### P2. Prompt recovery versus attribution

**Recovery is established, and closed-set attribution is explicitly formalized in the closest 2026 work.** *Extracting Prompts by Inverting LLM Outputs* introduces output2prompt, which reconstructs prompts from ordinary generated outputs without logits or jailbreak queries. *Reverse Prompt Engineering* studies limited-output, training-free reconstruction. These establish an inverse problem, not an assertion that the original string is uniquely recoverable.[^10][^11]

Let P be a prompt, Q a task, and T a trace. Recovery estimates a string or equivalence class, P-hat(T). Closed-set attribution instead returns a label J-hat(T) from a registered collection {p1,...,pK}. The latter can succeed without recovering any instruction words: a discriminative model needs stable differences between conditional output distributions. Recovery can also appear successful by producing a different prompt with similar behavior. String equality, semantic similarity, and functional equivalence are distinct recovery criteria.

Open-set matching asks whether two traces or sets of traces share a generating prompt, including prompts absent from training. This is different again from reading a previously unseen key text and selecting its outputs with no examples. A fixed K-way classifier cannot meaningfully predict labels excluded from its output vocabulary. A valid unseen-key experiment needs a defined text-conditioned scorer, a learned matching function, or a support-set prototype.

Two naming traps matter. XPrompt uses “prompt attribution” for explaining which input spans influence generation, not for identifying the source prompt. PromptCARE protects prompts through injected verification behavior, rather than passively recognizing arbitrary persona effects. PROMPRINT, an ACL 2026 first-token fingerprinting paper, is a relevant lead, but a complete primary paper was not retrieved here and its detailed results are **unverified**.[^12][^13][^45]

### P3. Function words, punctuation, and adversarial obfuscation

**Their attribution value is established; their universal resistance to rewriting is not.** The classic origin is Mosteller and Wallace’s Bayesian Federalist authorship study (1964), discussed in Kestemont’s account of function-word stylometry. Kacmarcik and Gamon already distinguish shallow from deeper anonymization and study changing the features an attribution model uses. These precedents undermine both “content-free features are surprising” and “such features are intrinsically ineradicable.”[^5][^14]

The Brennan-Afroz-Greenstadt experiments are seminal, but the replication by Wang, Juola, and Riddell is especially relevant to this project: it identifies an omitted control group and revises the interpretation of obfuscation effectiveness. Its ten-author experiments obtain roughly 20% attribution after manual interventions, against 10% chance. That is an empirical residual for particular attacks, not a universal lower bound.[^15]

Modern attacks include ALISON's interpretable stylometric perturbations, combinatorial paraphrasing, and personalized LLM rewriting. They demonstrate that “paraphrase” is not one intervention: local substitutions, sentence restructuring, an imposed target style, and classifier-guided search have different effects. Deobfuscation work further shows why testing only the original classifier can overstate removal; an attacker may fool a decision rule while leaving recoverable information.[^16][^17][^18][^19]

**No task-independent numerical ceiling or positive attacker floor exists for the stated problem.** If an attacker may ignore the source and return a fixed text, a balanced-key observer loses all key information. Under content constraints, the admissible rewrite set determines what remains. The important quantity is the best attainable attribution under a specified fidelity constraint and attacker budget, evaluated with multiple readouts and ideally a newly trained decoder.

An 8.3-point accuracy loss demonstrates the cost of one procedure. It does not establish the minimum cost of removing the signal. *Watermarks in the Sand* gives a conditional impossibility result under access to quality and perturbation oracles and suitable mixing; it does not prove that every practical paraphraser can erase every watermark at no cost.[^7]

### P4. Style, semantics, alignment, and activation directions

**There is operational decomposition, not a universally identifiable style/content factorization.** LIMA's superficial-alignment hypothesis concerns learning response formats with limited post-training data. It does not prove that arbitrary persona effects are shallow, harmless, or independent of computation. StyleDistance explicitly trains content-independent style embeddings using controlled paraphrase pairs; a generic semantic sentence encoder is therefore an incomplete comparator for a style task.[^20][^21]

Persona Vectors identifies activation directions associated with behavioral traits and intervenes on them. This supports the feasibility of causal experiments with internal representations. It does not establish that each natural-language key selects an arbitrary attractor or that different key phrasings share a unique vector. A 2026 preprint on steering-vector identifiability explicitly warns that behaviorally equivalent interventions need not identify one unique direction; its formal assumptions should be examined before importing its conclusions.[^22][^23]

For this project, an “attractor” or “basin” would require evidence of convergence from different initial conditions, stability under perturbation, and defined transition behavior. A cluster of output embeddings is insufficient. Prefer “prompt-conditioned distribution” until intervention data justify a stronger dynamical account.

The premise “function words and punctuation are content-free” is also too strong for GSM8K, MATH, and especially FOLIO. Negation, conditionals, quantifiers, causal connectives, parentheses, and equation punctuation can encode the argument itself. Call the feature set **restricted surface stylometry**, and separately test semantic and solution-structure dependence.

### P5. Detection, attribution, and capacity

**The distinction is established.** Zero-bit watermarking tests whether a mark is present; multi-bit watermarking encodes a recoverable message such as an identity. Codable Watermarking and Distributional Information Embedding directly address payloads and trade-offs among error, quality, and rate. Their control over generation distributions is stronger than natural-language prompt control, so their achievable rates cannot be assigned to this persona channel.[^24][^25]

Detection compares a null distribution P0 to a keyed mixture. Attribution must distinguish the component distributions P1,...,PK. For an elementary counterexample, let every keyed distribution equal the same P-star, while P-star differs from P0. Detection can be easy and key attribution exactly chance. Conversely, a mixture of distinguishable keyed distributions can equal P0: key information may be recoverable even when keyed-versus-null detection fails. These are statistical distinctions, not special properties of CoT.

A directly titled July 2026 preprint, *Watermark Forensics for Generative Models*, proposes a forensic ladder separating detection, attribution, payload extraction, and localization. Its indexed abstract is relevant to the proposed capacity story; full primary text was not accessible, so the report does not rely on its theorem details.[^26]

For a uniformly sampled key J, independent of task Q, and a trace T, Fano's inequality gives, in bits:[^54]

**I(J; T | Q) >= log2(K) - h2(Pe) - Pe log2(K-1).**

Here Pe is an attribution error probability, Q is available task context, and h2 is binary entropy. This is a necessary-information bound derived from a decoder's error, not a capacity estimate. An empirical confusion matrix also yields I(J; J-hat | Q), a decoder-dependent lower bound when estimated appropriately. “Eight keys carry three bits” is only the label entropy, not three reliably decoded bits.

If I(J; T | Q) is at most nC, the weaker form implies nC >= (1-Pe)log2(K)-1. Applying it requires a defensible C and a stated distribution over tasks; tokens are not independent samples merely because there are n of them. A K=64 sweep with no rate model, data-budget control, or fidelity constraint measures classifier scaling rather than channel capacity.

### P6. Negative results and length confounds

**The precise warning about trigger-text cosine and the stated AUROC collapse was not found.** Adjacent warnings are strong enough that the paper should present this as a controlled methodological result, without claiming a new general principle. Reasoning-data selection already has a specific step-length-confounding analysis; text-watermark power routinely depends on output length, entropy, and the evidence available to the detector.[^8][^27]

Do not conflate three embedding tests: similarity of trace to instruction, similarity between two traces, and a supervised classifier on trace embeddings. They answer different questions. DIPPER work, for example, finds semantic retrieval useful for recovering known generated passages after paraphrasing; it does not support the blanket claim that semantic readout fails.[^28]

Length is also a post-instruction variable. If the key changes verbosity, length carries real key information, even if it is undesirable for the intended application. Matching on length changes the estimand to attribution beyond length and can select a different difficulty distribution. Report length-only performance, randomized output budgets, within-question comparisons, matched subsets, and fixed-prefix views. A drop in one detector's AUROC does not establish that most information available to all detectors came from length.

## 3. Secondary connections and statistical formulation

### S1. Faithfulness and monitorability

Style variation with an unchanged answer neither establishes nor refutes CoT faithfulness. Multiple faithful descriptions can express the same computation; an unfaithful rationale can also preserve the final answer. Lanham et al. already intervene by paraphrasing CoTs to investigate dependence on their wording. Turpin et al. demonstrate unreported influences on answers, a different evidential test from detecting persona style.[^29][^30]

Monitorability asks whether the trace exposes relevant behavior, even if the trace is not a complete causal transcript. Baker et al. show that optimizing against a CoT monitor can induce obfuscation. The experiment suggested here is narrower: keep verifiable reasoning content fixed, randomize persona/style, and test whether a monitor's judgment changes. That would identify a nuisance dependency in oversight rather than infer faithfulness from attribution.[^31]

### S2. Agent propagation, provenance, and threat models

The closest work covers conversational-agent forensics, benign CoT verification behaviors for RAG ownership, and trained reasoning watermarks. R-CoT and BiCoT operate through model training or representations; they are important framing precedents, but do not establish prompt-only persona attribution after a chain of rewrites.[^1][^9][^32][^33]

Secret-collusion research formalizes senders, receivers, overseers, covert payloads, and mitigation. Hidden in Plain Text studies optimization-driven steganographic communication. Morris II demonstrates adversarial prompt replication and propagation between applications. These cover message-channel security, not trustworthy reconstruction of who contributed each reasoning step.[^34][^35][^36]

For this setting, report the attacker dimensions explicitly:

| Dimension | Required distinction |
| --- | --- |
| Attacker access | Only the candidate trace; original question; candidate keys; example keyed traces; detector label; detector score; generator weights |
| Rewrite constraint | Preserve answer; preserve all premises and conclusions; preserve each inferential step; preserve step order; preserve exact wording |
| Attack objective | Erase all attribution; evade one detector; impersonate another key; remove one ancestor while retaining another |
| Feedback and budget | One-shot instruction, best-of-N, iterative score queries, surrogate training, or gradient access; count queries/tokens |
| Relay memory | Previous trace only, cumulative trace history, original prompt retained, or original key reintroduced |
| Attribution target | Initial source, last writer, any ancestor, local span origin, or inferred solution strategy |

A prompt that says “destroy fingerprints” is **attack-aware rewriting**, but is not automatically a detector-adaptive attack. Adaptivity requires a specified feedback or model-access mechanism. Statistical resemblance also does not authenticate provenance: someone who knows a style can imitate it, and independent writers can generate similar traces. Signed logs can bind exact artifacts to an execution record; stylistic attribution supplies probabilistic evidence about unlogged or transformed artifacts.

### S3. Calibrated attribution on dependent, short text

Use whole questions or independent generation episodes as the primary sampling units. Tokens, several generations from one question, and successive hops in one lineage are dependent. A question-level split is valuable, but uncertainty estimation and permutation tests must respect that same clustering.

**Recommended construction.** Freeze the trained readout and calibrate on separate questions. For unkeyed/null traces, calibrate the maximum of the K attribution scores; threshold that maximum to control the chance of assigning any registered key under the null. Alternatively, construct valid per-key null p-values and use a familywise correction. A per-key false-positive rate is not the probability of any false accusation over K candidates.

For attribution conditional on belonging to the registered population, use class-conditional conformal prediction to return a set of compatible keys. The guarantee concerns inclusion of the true class under the stated exchangeability conditions, not the correctness of every singleton conditional on selection. Unknown-key rejection needs representative unkeyed and unregistered-key calibration data; ordinary multiclass conformal coverage does not solve arbitrary out-of-distribution detection.[^37]

Conformal p-values require exchangeable calibration and test units, or a justified extension. Exchangeable questions can contain arbitrarily dependent tokens. By contrast, a rewritten trace from a new attack distribution is not automatically exchangeable with clean calibration traces. Separate calibration by deployment regime and test calibration transfer explicitly.

For sequential evidence, valid likelihood-ratio processes or e-processes permit optional stopping when their conditional null guarantees hold. Multiplying p-values or e-values obtained independently from each hop is invalid when the necessary dependence conditions fail. Prefer fresh, independently sampled task episodes for evidence accumulation; otherwise model the full joint history.[^38]

Higher criticism and related sparse-signal tests are relevant when surviving watermark evidence occupies a small fraction of positions. Li et al.'s statistical-watermark framework and robust-under-edits work give principled detection rules and detection boundaries. Their key-dependent pivotal statistics do not come for free with a learned logistic score. For dense stylistic shifts, a calibrated multivariate score is a more natural first tool.[^39][^40]

At a target tail probability alpha, an empirical conformal p-value has resolution about 1/(m+1) with m calibration units. With zero observed false positives, a rough independent-binomial 95% upper bound is 3/m. Thus extreme forensic error-rate claims require thousands of independent negative cases, not thousands of tokens from a few cases. Report power and precision at the relevant prevalence as well as AUROC.

### S4. Radioactivity, membership inference, and dataset marking

Radioactive-data methods ask whether a model was trained on marked examples. Watermarking Makes Language Models Radioactive extends that idea to training on watermarked synthetic text. The relevant chain is marked data -> parameter update -> future outputs, whereas the present chain is prompt -> trace -> rewritten trace with no parameter update.[^41][^42]

Membership inference concerns whether a particular example belonged to a training set. Prompt attribution concerns which condition generated an output. Neither a successful prompt classifier nor a rewrite-resistant style proves training membership. A useful future bridge would test whether fine-tuning on persona-conditioned traces transmits their residual style after the original instruction is removed, but that becomes a new radioactivity experiment and needs its own controls.

The common information-theoretic principle is contraction: a transformation cannot increase information about the key unless additional key-correlated inputs enter. This supplies a clean check on multi-hop claims. Flat observed accuracy can coexist with decreasing information because the classifier may not use everything present in the trace.

### S5. Backdoors, reconstruction, and trigger scrubbing

BadChain establishes that manipulated CoT demonstrations can create trigger-conditioned unwanted reasoning behavior without parameter access. Hidden Killer and Mind the Style of Text establish nonliteral syntactic or stylistic triggers. Literal token deletion is therefore not a universal trigger defense.[^43][^44][^46]

ONION removes suspicious words using a perplexity criterion. It addresses a different object from this project's residual signal: sanitizing an input trigger before generation versus erasing information from an already generated output. A defense can prevent reactivation while leaving source-attributable language intact, or erase style while preserving an actionable malicious instruction.[^47]

Trigger reconstruction is likewise not ordinary provenance attribution: it searches for an input that activates specified behavior, which need not be the historical trigger. SemInv explicitly addresses textual trigger inversion, including semantic constraints and stealthy triggers, and distinguishes finding a trigger from deciding that a model is backdoored.[^53] A verified study showing that a particular scrubber necessarily removes the proposed idiosyncratic CoT channel was **not found**. Test input sanitization, output neutralization, and behavior prevention as separate endpoints.

### S6. Translation, forensic linguistics, and human rewriting

Translationese demonstrates that transformed language can preserve information about its source while acquiring regularities from the transformation. Koppel and Ordan distinguish translated English and its source-language varieties. This is a useful counterexample to treating post-transformation attribution as proof of source-word survival.[^48]

Baker's translator-style work is an older conceptual origin; its publication metadata was verified, but full text was unavailable here. It should be cited for the translator-style question, not for unverified numerical robustness claims. Human crowdsourced rewriting has also been explicitly evaluated as an authorship-linkability defense.[^49][^50]

The important model is a mixture of source, translator/rewriter, genre, and task effects. Successive-rewriting studies in LLMs directly connect to this tradition. The correct experiment crosses original keys with rewriter models and rewriter instructions, allowing both source identity and rewriter identity to be measured. A changing signature may reflect a new authorial contribution rather than simple destruction of the old one.

## 4. Annotated bibliography by community

Entries identify the result relevant here and delimit the overlap. “Full text checked” means relevant methods/results were available; it does not mean independently reproduced. “Abstract checked” supports only the stated high-level result. Venue labels are conservative when proceedings confirmation was unavailable.

### Watermarking and ownership verification

- **[3] Liu et al., In-Context Watermarks for Large Language Models. ICLR 2026; preprint 2025, revised April 2026. Full text checked.** Prompt-only marking uses Unicode, word initials, lexical preferences, and sentence acrostics, with tailored detectors and paraphrase/attack evaluations. **Overlap: direct for instructional embedding; partial for robustness; does not identify a latent persona channel.**
- **[4] Dasgupta, Tanvir, and Zhong, Watermarking Language Models through Language Models. Preprint 2024, revised 2025; publication venue not verified. Full text checked.** A prompting model creates watermark instructions and a classifier recognizes marked outputs; instruction types include linguistic constraints. **Overlap: prompt-driven linguistic distribution shifts, not the claimed causal two-channel decomposition.**
- **[7] Zhang et al., Watermarks in the Sand. ICML 2024. Full text checked.** Quality-preserving perturbation oracles with suitable mixing undermine strong watermark robustness. **Overlap: directly constrains interpretation of an attack plateau; theorem assumptions do not automatically hold for a practical CoT rewriter.**
- **[9] Guo et al., Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Reasoning. Preprint 2025, revised May 2025; venue not verified. Full text checked.** Distinct benign CoT verification behavior supports knowledge-base ownership testing while retaining correct answers. **Overlap: a direct threat to novelty of answer-preserving reasoning fingerprints, with a different insertion mechanism.**
- **[24] Wang et al., Towards Codable Watermarking for Injecting Multi-bits Information to LLMs. ICLR 2024. Abstract and venue checked.** Watermarks encode identities or other multi-bit metadata. **Overlap: detection versus payload attribution and coding efficiency; not prompt-only channel capacity.**
- **[25] He et al., Distributional Information Embedding. Preprint 2025; venue not verified. Abstract checked.** Formalizes quality, detectability, and information rate, including finite-token non-i.i.d. formulations. **Overlap: the right general capacity vocabulary; assumes stronger distributional control than persona prompting.**
- **[27] Kuditipudi et al., Robust Distortion-free Watermarks for Language Models. Preprint 2023; venue not independently verified here. Abstract checked.** Keyed sampling provides statistically testable marks; short, low-entropy instruction outputs are harder in their experiments. **Overlap: evidence-budget and entropy limitations, not a length confound in cosine detection.**
- **[32] Zhang et al., R-CoT. Preprint, April 2026. Full text checked.** GRPO training installs a redundant reasoning policy alongside ordinary reasoning. **Overlap: reasoning-path watermarking; excludes a claim that all reasoning marks must alter final answers; requires training.**
- **[33] Lu et al., Echoes within the Reasoning / BiCoT. Preprint, May 2026. Full text checked.** Embeds signatures in reasoning representations and verifies using top-logprob information. **Overlap: mechanistic coupling of marks to reasoning; stronger access assumptions than output-only prompt attribution.**
- **[51] McMichael, Roussinov, and Sharoff, Discourse-level Watermarking using Rhetorical Structure Theory. IEEE ICNLP 2026. Institutional publication record/abstract checked.** Uses rhetorical relations in a black-box prompt-based pipeline and reports improved paraphrase robustness. **Overlap: discourse structure as a deliberately designed carrier; a close D3/D4 baseline, with no verified persona mechanism.**

### Stylometry, authorship, and translation

- **[5] Kestemont, Function Words in Authorship Attribution: From Black Magic to Theory? CLfL 2014. Full text checked.** Examines why frequent grammatical features can identify authors and questions simplistic feature explanations. **Overlap: directly establishes the feature principle behind claim 3; does not establish paraphrase immunity.**
- **[6] Tripto et al., A Ship of Theseus. ACL 2024. Full text checked.** Repeated LLM paraphrasing changes style and authorship-classification behavior. **Overlap: direct precedent for studying iterative transformations; does not establish a ten-hop key plateau.**
- **[14] Kacmarcik and Gamon, Obfuscating Document Stylometry to Preserve Author Anonymity. COLING/ACL 2006 poster proceedings. Full text checked.** Alters identifying features and distinguishes shallow from deep anonymization. **Overlap: direct predecessor of feature-targeted style removal.**
- **[15] Wang, Juola, and Riddell, Reproduction and Replication of an Adversarial Stylometry Experiment. Preprint 2022, revised March 2026; venue not verified. Full text checked.** Replicates Brennan et al. (ACM TISSEC 2012) while correcting missing controls. **Overlap: evidence that an apparent residual is attack/protocol dependent; the older original was not independently retrieved.**
- **[16] Xing et al., ALISON. AAAI 2024. Full text checked.** Fast interpretable authorship obfuscation works without direct target-classifier signals during rewriting. **Overlap: a concrete stronger comparator to generic “destroy style” instructions.**
- **[17] Gröndahl and Asokan, Effective Writing Style Imitation via Combinatorial Paraphrasing. PoPETs 2020. Abstract/publication record checked.** Searches combinations of paraphrases to imitate target styles. **Overlap: targeted spoofing and a content-preserving attack framework.**
- **[18] Shokri, S. I. Levitan, and R. Levitan, Personalized Author Obfuscation with Large Language Models. Preprint 2025; venue not verified. Abstract checked.** Obfuscation efficacy differs substantially across authors; personalized prompts improve it. **Overlap: predicts key-specific attack vulnerabilities hidden by mean accuracy.**
- **[19] Zhai et al., A Girl Has A Name, And It's ... Adversarial Authorship Attribution for Deobfuscation. ACL 2022. Abstract checked.** Studies attribution after obfuscation. **Overlap: motivates retrained decoders, rather than equating a frozen detector's failure with information removal.**
- **[21] Patel et al., StyleDistance. NAACL 2025. Abstract and venue checked.** Controlled synthetic paraphrases train representations that distinguish style while reducing content dependence. **Overlap: essential comparator for claim 2 and a direct operational precedent for style/content separation.**
- **[48] Koppel and Ordan, Translationese and Its Dialects. ACL 2011. Primary record checked.** Investigates statistical signatures of translation and original-language influence. **Overlap: source and transformation can both leave attributable effects; not repeated CoT paraphrasing.**
- **[49] Baker, Towards a Methodology for Investigating the Style of a Literary Translator. Target 12(2), 2000. Metadata only; full text unavailable.** Establishes the translator-style research question. **Overlap: conceptual origin of translator fingerprints; no quantitative survival claim is used here.**
- **[50] Almishari, Oguz, and Tsudik, Fighting Authorship Linkability with Crowdsourcing. ACM COSN 2014. Abstract checked.** Evaluates human rewriting for reducing authorship linkability. **Overlap: a human transformation baseline and warning against assuming rewriting merely adds harmless noise.**

### Prompt attribution and inversion

- **[1] White, Jafari, and Berg-Kirkpatrick, Black-Box Forensics for Conversational LLM Agents. Preprint, June 2026. Full text checked.** Closed-set prompt attribution uses lexical/stylometric readouts; open-set matching uses learned encoders. **Overlap: strongest direct novelty threat to the basic attribution framing; no exact experiment on this key set or ten-hop CoTs.**
- **[2] L. Chen et al., Do System Prompts Leave Behavioral Fingerprints? Preprint, August 2026. Indexed primary abstract only.** Reports behavioral clone detection, prompt-paraphrase robustness, and tone-prefix vulnerability. **Overlap: direct at a high level; detailed controls remain unverified.**
- **[10] C. Zhang, Morris, and Shmatikov, Extracting Prompts by Inverting LLM Outputs. EMNLP 2024. Abstract and publication attribution checked.** output2prompt recovers prompts from ordinary outputs without logits. **Overlap: prompt information in outputs, with a generative recovery objective rather than a finite codebook.**
- **[11] H. Li and Klabjan, Reverse Prompt Engineering. Preprint 2024; venue not verified. Abstract checked.** Recovers coherent prompts with limited outputs and no task-specific inversion training. **Overlap: functional-equivalence baselines and limits of string-exact recovery.**
- **[12] Chang et al., XPrompt. Preprint 2024; venue not verified. Abstract checked.** Attributes generation to combinations of input spans using counterfactual explanations. **Overlap: vocabulary collision, not provenance attribution.**
- **[13] Yao et al., PromptCARE. Preprint 2023; venue not verified here. Abstract checked.** Protects prompts through watermark injection and verification. **Overlap: active prompt ownership protection, distinct from naturally induced persona signatures.**
- **[45] Lee, Lv, and Lee, PROMPRINT: Prompt Fingerprinting via First-Token Response for LLM App Cloning Detection. ACL 2026 listing found; full paper not retrieved.** **Unverified lead:** app-clone detection using first-token responses. Do not treat it as evidence for a specific stylometric result without reading the paper.

### CoT faithfulness and model behavior

- **[8] B. Wang et al., On the Step Length Confounding in LLM Reasoning Data Selection. Preprint, April 2026; proceedings status not independently verified. Full text checked.** Naturalness-based reasoning-data scores can confound quality with step length. **Overlap: a methodological predecessor, not the same attribution AUROC collapse.**
- **[20] Zhou et al., LIMA: Less Is More for Alignment. NeurIPS 2023. Abstract checked.** Limited curated post-training examples support strong response-format learning. **Overlap: motivates shallow-format hypotheses, without proving style/content independence.**
- **[22] R. Chen et al., Persona Vectors. Preprint 2025; venue not verified. Abstract checked.** Extracts and intervenes on trait-associated activation directions. **Overlap: a tool for causal tests, not evidence of arbitrary style basins.**
- **[23] Venkatesh and Kurapath, On the Identifiability of Steering Vectors in Large Language Models. Preprint, February 2026. Primary full text available; theorem assumptions not exhaustively audited.** Studies non-uniqueness of behaviorally similar steering interventions. **Overlap: caution against equating one successful direction with a unique mechanism.**
- **[29] Lanham et al., Measuring Faithfulness in Chain-of-Thought Reasoning. Preprint 2023. Abstract checked.** Intervenes through mistakes, paraphrases, and other changes to assess answer dependence on CoT. **Overlap: directly precedes the style-versus-answer question; not prompt provenance.**
- **[30] Turpin et al., Language Models Don't Always Say What They Think. NeurIPS 2023. Abstract and venue checked.** Biasing context can affect answers without acknowledgment in the rationale. **Overlap: causal faithfulness, not stylistic similarity.**
- **[31] Baker et al., Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. Preprint 2025. Abstract checked.** CoT monitoring detects reward hacking; optimization against it can promote concealment. **Overlap: monitor-aware behavior, with a different optimization and threat model.**
- **[52] D. Lee, Queen, and Zou, ReasonOps. Preprint, May 2026. Full text checked.** Sentence-initial pivots and discourse operators identify source models and predict correctness. **Overlap: especially close to D3's step-opening features; model attribution rather than key attribution.**

### Agent security

- **[34] Motwani et al., Secret Collusion among AI Agents. NeurIPS 2024. Abstract/publication record checked.** Formalizes covert communication and model capabilities for agent collusion. **Overlap: sender/receiver/overseer threat models and channel capacity, not authentic lineage reconstruction.**
- **[35] Mathew et al., Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs. Preprint 2024; venue not verified. Primary text available.** Optimization can produce covert communication without directly prescribing a hand-built code. **Overlap: emergent nonliteral channels and removal defenses, not the measured persona mapping.**
- **[36] Cohen, Bitton, and Nassi, Here Comes the AI Worm / Morris II. Preprint 2024; venue not verified. Abstract checked.** Studies adversarial self-replicating prompts propagating through connected applications. **Overlap: multi-hop propagation and re-exposure controls; measures security behavior, not residual stylometry.**
- **[28] Krishna et al., Paraphrasing Evades Detectors of AI-Generated Text, but Retrieval Is an Effective Defense. NeurIPS 2023. Abstract checked.** DIPPER defeats several detectors; retrieval of known generated text helps recover evidence. **Overlap: both an attack baseline and a counterexample to dismissing all semantic readouts.**

### Statistical detection theory and radioactivity

- **[26] X. Li et al., Watermark Forensics for Generative Models: An Information-Theoretic Perspective. Preprint, July 2026. Indexed primary abstract only.** Separates forensic objectives and relates attribution to a tokenwise information profile. **Overlap: highly relevant to P5/D2; theorem details unverified.**
- **[37] Bates et al., Testing for Outliers with Conformal p-values. Annals of Statistics 51(1), 2023; preprint 2021. Abstract checked.** Constructs calibrated outlier p-values and studies their dependence and multiple testing. **Overlap: null calibration, not an automatic solution to unknown attacks.**
- **[38] Ramdas, Grünwald, Vovk, and Shafer, Game-theoretic Statistics and Safe Anytime-valid Inference. Statistical Science, 2023; preprint 2022. Primary record checked.** Explains e-values, e-processes, and optional-stopping-safe inference. **Overlap: sequential evidence under valid null conditions.**
- **[39] X. Li et al., A Statistical Framework of Watermarks for Large Language Models. Annals of Statistics 53(1), 2025; preprint 2024. Abstract/publication record checked.** Pivotal statistics support false-positive control and efficient detection. **Overlap: the required separation of calibration and power; not a guarantee for arbitrary learned scores.**
- **[40] X. Li et al., Robust Detection of Watermarks for Large Language Models under Human Edits. Preprint 2024, revised 2025; venue not verified. Abstract checked.** Studies detection after edits and sparse surviving signals. **Overlap: sparse-evidence methods and their assumptions, rather than generic stylometric invariance.**
- **[41] Sablayrolles et al., Radioactive Data: Tracing through Training. ICML 2020. Abstract checked.** Marked training images leave statistically detectable effects in trained models. **Overlap: conceptual origin of dataset radioactivity; a different information path.**
- **[42] Sander et al., Watermarking Makes Language Models Radioactive. NeurIPS 2024. Abstract/publication record checked.** Detects training on watermarked synthetic text through residual watermark evidence. **Overlap: a possible future prompt-style distillation study, not present-tense trace provenance.**
- **[54] Scarlett and Cevher, An Introductory Guide to Fano's Inequality with Applications in Statistical Estimation. Preprint chapter, 2019. Abstract checked.** Develops information-theoretic lower-bound methods for multi-hypothesis estimation. **Overlap: a mathematical reference for capacity limitations; not an achievable-rate result for text prompting.**

### Backdoors

- **[43] Xiang et al., BadChain. ICLR 2024. Abstract checked.** Demonstration-level CoT poisoning creates trigger-conditioned malicious reasoning without changing model parameters. **Overlap: prompt-conditioned reasoning behavior; the intended outcome differs from benign attribution.**
- **[44] Qi et al., Hidden Killer. ACL-IJCNLP 2021. Abstract checked.** Syntactic structure can serve as a nonliteral backdoor trigger. **Overlap: token scrubbing may miss distributed structure; input trigger rather than output signature.**
- **[46] Qi et al., Mind the Style of Text! EMNLP 2021. Abstract checked.** Style transfer produces adversarial examples and backdoor triggers while preserving meaning. **Overlap: direct precedent for task-irrelevant style as a security channel.**
- **[47] Qi et al., ONION. EMNLP 2021; preprint 2020. Abstract checked.** Removes suspicious input words using perplexity changes. **Overlap: a lexical trigger defense, without a guarantee of removing post-generation persona fingerprints.**
- **[53] Xie et al., Semantic and Precise Trigger Inversion: Detecting Backdoored Language Models. IEEE TIFS 20, 2025. Abstract and introduction checked.** SemInv combines semantic constraints on inverted triggers with a separate backdoor-determination criterion, including stealthy-trigger tests. **Overlap: direct trigger-reconstruction precedent; no claim about erasing benign output fingerprints.**

## 5. Vocabulary map

| Community | Terms to search | What the term usually means | Search trap |
| --- | --- | --- | --- |
| Prompt forensics | system-prompt attribution; behavioral fingerprinting; prompt clone detection; closed-set prompt identification | Infer a deployed prompt or match endpoints from their outputs | “Prompt fingerprinting” can also mean an optimized query for model identification |
| Inversion/security | output2prompt; language-model inversion; prompt reconstruction; functional prompt recovery; system-prompt extraction | Reconstruct hidden instructions or an equivalent prompt | Recovery need not identify the historical original string |
| Explainability | input attribution; joint prompt attribution; counterfactual prompt explanation | Identify influential input spans | Usually not source attribution |
| Stylometry | writeprint; idiolect; authorship verification; authorship attribution; cross-topic attribution | Identity from recurring linguistic behavior | Function words are not guaranteed independent of topic or reasoning content |
| Privacy | authorship obfuscation; author masking; style imitation; deobfuscation; authorship linkability | Hide, spoof, detect masking, or recover identity | Attacking a frozen classifier is not proving identity information is absent |
| Watermarking | zero-bit detection; multi-bit watermark; in-context watermark; payload extraction; false accusation; collusion resistance | Encode and verify a chosen signal or identity | A textual “key” need not be a cryptographic secret |
| Linguistic watermarking | morphosyntactic watermark; acrostic; discourse watermark; rhetorical relations | Deliberately control linguistic structure | Robustness of one carrier does not transfer to another |
| Representation research | content-independent style embedding; contrastive style representation; persona vector; activation steering | Operationally isolate or manipulate stylistic/behavioral variation | A successful probe or steering vector does not uniquely identify a mechanism |
| CoT research | rationale faithfulness; intervention faithfulness; monitorability; reasoning operators; discourse pivots | Relation of traces to answers, behavior, and internal processing | Accurate provenance classification says little by itself about faithful reasoning |
| Agent security | covert channel; steganographic collusion; overseer; semantic firewall; indirect prompt injection | Unauthorized information flow or instruction propagation | Relay of an original key is reactivation, not evidence that a trace alone retained it |
| Statistics | composite hypothesis; multi-hypothesis testing; selective classification; conformal prediction set; e-process; sparse mixture | Calibrated decisions with nuisance parameters and repeated looks | Marginal calibration does not imply conditional or distribution-shift guarantees |
| Information theory | Fano bound; Chernoff information; channel resolvability; rate-distortion; privacy funnel; information profile | Distinguishability and information budgets under constraints | K labels and above-chance accuracy are not a capacity theorem |
| Data provenance | radioactivity; dataset ownership verification; membership inference; watermarked canary | Detect training/retrieval use of protected data | Training provenance differs from inference-time conditioning |
| Translation studies | translationese; translator style; translator fingerprint; source-language interference; explicitation | Signatures of source and transformer in translated text | A surviving signature may be structural transfer rather than source-word copying |

High-yield query combinations include **system prompt attribution + paraphrase**, **authorship obfuscation + deobfuscation**, **in-context watermark + syntax**, **reasoning operators + model identification**, **translator style + source interference**, and **watermark + false accusation + multi-bit**. Search “paraphrase of the instruction” separately from “paraphrase of the output.”

## 6. Three strongest threats to novelty or interest

**Threat 1: ordinary prompt attribution already explains the central observation.** The White et al. study is close enough that adding CoT datasets alone risks looking like a domain extension. The necessary distinction is an explanation of *which variation is induced, how it is maintained, and when it changes reasoning*, rather than another successful classifier.[^1]

**Threat 2: combining prompt-only marks with benign reasoning marks is already an available design path.** ICW supplies prompt-level insertion; Guo et al. supplies answer-preserving CoT verification; ReasonOps supplies interpretable reasoning-style features. A paper whose main contribution is a convenient combination faces a strong incrementalism objection. A causal interaction or a demonstrable constraint on controllability would be more substantial.[^3][^9][^52]

**Threat 3: the robustness story can collapse under a better-specified rewriter.** Successive-paraphrase attribution and adversarial style manipulation are established. A generic repeated rewrite can preserve the same structure for ten hops, while constrained regeneration removes it in one. The observed trade-off is not yet a frontier and the plateau is not a bound. This threatens practical interest even if the dataset-specific curve is new.[^6][^7][^16]

The August behavioral-fingerprint and July forensic-information preprints are additional urgent leads because their stated topics are extremely close. Their incomplete accessibility limits the strength of conclusions drawn from them, not the need to compare against them before submission.[^2][^26]

## 7. Research directions ranked by decision value

The ranking favors experiments that could kill or materially revise the interpretation quickly. Novelty judgments below refer to the precise experiment; they are not guarantees that a publishable gap exists.

| Rank | Direction | Recommendation |
| --- | --- | --- |
| 1 | D1: Paraphrases and minimal edits of the same key | **Do first, in strengthened form.** Prompt paraphrase effects already studied; identify semantic and wording components jointly. |
| 2 | N1: Residual signal after a verified semantic bottleneck | **Highest-value new falsifier.** Distinguishes inherited wording, solution choice, and genuine residual dependence. |
| 3 | N2: Continuous prompt influence versus prefix self-conditioning | **Strong mechanistic candidate.** Test where and how a fingerprint persists during generation. |
| 4 | D3: Causal carriers of the surface signature | **Worth doing as interventions.** Descriptive feature rankings alone are occupied. |
| 5 | N3: Style interventions on downstream reasoning and monitors | **Strong behavior/safety reframe.** Test consequences, not merely recognizability. |
| 6 | D4: Independent channel controls and a usable frontier | **Conditional on D1/D3.** Strong only if separation predicts useful behavior under held-out attacks. |
| 7 | N4: Origin versus last-writer identity in trace genealogy | **Useful if the agent application is real.** Resolve what attribution actually refers to. |
| 8 | D2: K=8 to K=64 channel scaling | **Supporting study.** A raw accuracy sweep is a weak primary contribution. |
| 9 | N5: Privacy leakage of hidden prompt attributes | **Optional behavior reframe.** Valuable if it uncovers protected information beyond public stylistic instructions. |

### D1. Do equivalent key phrasings induce equivalent fingerprints?

**Question and importance.** Is the signature stable across semantically equivalent instructions, or tied to particular lexical realizations? This determines whether “idiosyncratic” describes the model's response to meaning, incidental wording, or an inadequate semantic representation. Prior prompt-paraphrase studies partially answer the broad question; the controlled decomposition remains the opportunity.[^1][^2]

**Experiment.** For each of eight semantic personas/instructions, create at least four independently written paraphrases, plus matched minimal edits that change meaning with little lexical change and edits that change surface form while preserving meaning. Independently validate instruction equivalence. Cross every form with the same questions and multiple generation seeds; hold out both questions and instruction phrasings. Fit nested models for persona family and wording-within-family, using lexical, restricted surface, StyleDistance, and text-conditioned readouts. Evaluate original-to-paraphrase transfer against within-form repeatability, not only against chance.

**Falsification.** If cross-paraphrase transfer is equivalent to within-form transfer within a preregistered margin, while semantic changes reliably move the fingerprint, the strong wording-idiosyncratic hypothesis fails. If equivalent phrasings separate as strongly as distinct personas, a stable semantic-codebook interpretation fails. A mixed result requires variance decomposition, not choosing whichever binary framing is more attractive.

### D2. What is the channel's effective capacity as K grows?

**Question and importance.** How many keys can be distinguished at a stated trace budget, error tolerance, and fidelity level? Existing coding theory answers the general question; the prompt-access constraint and measured stylistic channel could supply a new empirical regime.[^24][^25]

**Experiment.** Sample multiple independent codebooks at K=8,16,32,64. Compare fixed total training budget with fixed samples per key, and fix output-token budgets. Separate persona-only, explicit lexical, and combined keys. Optimize codebooks on development questions only; evaluate held-out questions, key phrasings, and rewriters. Report accuracy, calibrated rejection, confusion entropy, bits decoded per trace, learning curves, and task quality. Include a content-independent style encoder and a white-box conditional-likelihood scorer where available.

**Falsification.** If apparent saturation disappears with training data or a better decoder, it is not evidence of an intrinsic channel limit. If gains require overt identifiers, large quality loss, or a hand-picked codebook, the usable idiosyncratic-capacity hypothesis fails. A K-sweep alone should not lead the paper.

### D3. What linguistic features causally carry the signal?

**Question and importance.** Are sentence rhythm, connectives, punctuation, and step openings carrying independent information, or proxying for different solution plans? These are established stylometric features, and ReasonOps is particularly close for reasoning pivots.[^5][^52]

**Experiment.** Compare controlled renderings of one verified solution: normalize punctuation, standardize step openings, equalize sentence lengths, substitute connectives only where logically valid, and finally render an identical symbolic plan. Include a sham rewrite of matched magnitude. Use both a frozen classifier and classifiers retrained on each transformed distribution. Quantify residual key information beyond task, answer, length, and the verified solution representation; validate the transformations' logical fidelity.

**Falsification.** If feature importance does not predict selective intervention effects, the proposed carrier explanation fails. If equivalent information returns under a new decoder, the intervention disabled the readout rather than removed the signal. If the signal disappears only when solution strategies are equated, call it a strategy-conditioned signature rather than pure surface style.

### D4. Can independently designed channels yield a useful frontier?

**Question and importance.** Can two knobs alter literal expression and residual style independently, providing a predictable choice between detectability, robustness, and quality? Prompt-based lexical/discourse watermarking already covers much of the engineering motivation.[^3][^4][^51]

**Experiment.** Use a crossed 2x2 design: explicit lexical constraint present/absent and persona constraint present/absent, with balanced prompt length and randomized task pairing. Vary strength within each factor. Test lexical deletion, structural normalization, style transfer, and verified regeneration under equal attack budgets. Estimate main effects, interactions, and each readout's incremental information. Construct the Pareto frontier on development data; require its ordering to predict performance under held-out rewriters and tasks.

**Falsification.** If one factor merely changes compliance with the other, both vanish under the same intervention, or a single latent factor explains their performance, the independently controllable two-channel claim fails. If one setting dominates all others or the ranking reverses across rewriters, there is no stable practitioner trade-off of the proposed kind.

### N1. Does the residual survive a verified semantic bottleneck?

**Question and importance.** Is the plateau persistent source style, preserved solution structure, or repeated copying? This is a sharper model-behavior question than measuring an eleventh rewrite. It also probes the assumptions behind strong-removal impossibility results.[^7]

**Experiment.** Compare four transformations: ordinary paraphrase; fixed-template expression of an extracted solution graph; regeneration from a verified graph in a fresh context; and solving anew from the question alone. For GSM8K/MATH, use checked operations/equations; for FOLIO, validated predicates and inference relations. Preserve the same solution in the strict condition and permit alternative valid solutions in a separate condition. Randomize rewriter and seed, remove key/history access, and verify graph sufficiency and answer fidelity.

**Falsification.** If fresh rendering of the same solution erases attribution without material quality loss, the robust residual-style interpretation fails. If attribution survives a supposedly key-independent bottleneck, first test whether the graph retains key-specific strategy/order or whether metadata leaked; it does not establish information creation. If only independent re-solving removes attribution, the carrier may be solution choice, not superficial style.

### N2. Is the fingerprint maintained by the key or by the generated prefix?

**Question and importance.** Once a style begins, does the model continue it through autoregressive imitation even after the initiating condition is removed? This distinguishes continuous prompt conditioning from a self-maintaining generated trajectory. Persona steering provides intervention tools but does not answer this persistence question.[^22]

**Experiment.** At several prefix lengths, construct all four continuations: key A/prefix A, key A/prefix B, key B/prefix A, key B/prefix B. Also remove the key and regenerate from each prefix. Keep task and verified intermediate state aligned. Recompute the full context/KV state when changing the prompt; simply masking a prompt while retaining its cached influence is not a clean intervention. Measure attribution along the continuation, answer quality, and matched activation interventions against norm-matched random directions.

**Falsification.** If removing the key eliminates the effect immediately and swapped prefixes have no sustained effect, prefix self-maintenance fails. If switching the key cannot redirect behavior after controlling prefix semantics, a continuous-conditioning account fails. Broad quality collapse from activation removal is not evidence for a specific style mechanism.

### N3. Can style alter a recipient's reasoning or a monitor's decisions?

**Question and importance.** Does persona-conditioned presentation change how another agent uses an otherwise equivalent solution? This turns a provenance observation into a test of reasoning and oversight invariance. Existing faithfulness and monitor-obfuscation studies supply the comparison, but do not settle this matched-content intervention.[^29][^31]

**Experiment.** Present verified equivalent traces in crossed styles to a recipient that must solve a follow-up, locate an inserted error, or judge whether the rationale supports the answer. Keep all task-relevant premises fixed; include paired valid and invalid traces. Test monitor calibration and recipient error propagation with style randomized independently of correctness. Include recipients and monitors absent from prompt/attack tuning.

**Falsification.** If outcome and monitoring differences stay within a predefined equivalence margin, the claimed consequential style effect fails even if attribution remains high. If effects disappear after correcting premise loss or length imbalance, they are not evidence for stylistic manipulation of reasoning.

### N4. Whose fingerprint remains in a rewritten reasoning lineage?

**Question and importance.** Can an observer distinguish original source, latest rewriter, and mixed ancestry? This matters because an attribution label is otherwise ambiguous. Successive authorship and translation research supply the conceptual precedent.[^6][^48]

**Experiment.** Cross source keys with rewriter models and instructions. Include source-only relays, newly keyed rewrites, unkeyed rewriters, and synthesis from two independently keyed traces. Evaluate source identity, last-writer identity, and any-source set prediction separately. In controlled splicing conditions, test local-span localization; in free synthesis, use ancestry labels rather than pretending every output sentence has a unique source.

**Falsification.** If predictions mostly follow the latest rewriter, the method cannot claim initial-origin provenance. If mixtures force unstable single labels and set prediction offers no calibrated coverage, the single-source application does not extend to collaborative synthesis. No genealogy should be inferred from a classifier that was trained only on single-parent traces.

### N5. Does a hidden persona leak private prompt attributes beyond the intended task?

**Question and importance.** A stylistic signal useful for provenance can be an unintended information leak. The relevant question is whether hidden attributes are inferable without explicit mentions and beyond a known-key training set. Prompt inversion supplies the broad precedent; a privacy result must add a clearly distinct attribute and threat model.[^10][^11]

**Experiment.** Use synthetic private attributes that are deliberately irrelevant to the task. Separate instruction identity, semantic attribute, and incidental wording in a crossed design. Train on one set of phrasings and test unseen phrasings/values where the prediction task is well-defined. Compare literal redaction, neutral-style instructions, and the verified bottleneck. Measure utility and calibrated attribute inference, not only K-way key accuracy.

**Falsification.** If inference depends on overt mentions, memorized key labels, or correlations with the task, the proposed covert-attribute leakage fails. If preserving task utility necessarily preserves an explicitly requested role, the setup is not evidence of an unintended leak; redesign the attribute manipulation.

## 8. Minimum decisive study and claim discipline

A compact first study should combine **D1 + N1 + one D3 intervention**. Use the same question/key pairs so differences are paired, and reserve an untouched confirmation split. Choose sample size from the desired confidence interval and a preregistered equivalence margin, not from the number of inexpensive generations that fit the budget. The first goal is to determine whether the strong hypothesis survives, not to scale every configuration immediately.

The minimum analysis should report raw percentages alongside chance-normalized measures; per-key and per-question uncertainty; length-only and fixed-budget controls; train-only feature fitting; null results under appropriately clustered permutations; unknown-key/null rejection; and both frozen and retrained decoders after attacks. State whether an attack knows keys, sees scores, or was selected after examining test performance. Keep answer preservation separate from preservation of the full derivation.

Three interpretation rules are especially important. First, top-feature overlap of 0.11-0.16 is not “essentially zero” without a matched chance baseline accounting for instruction length, feature vocabulary, and selection. Second, no effect from changing a maximum pairwise cosine is not equivalence evidence for the entire codebook's geometry. Third, different cross-model n-gram overlap values do not causally identify instruction-following ability when model family, tokenizer, size, and prompting protocol also differ.

The defensible present claim is: **under a fixed candidate key set, specified generators, and the tested transformations, generated reasoning retains statistically decodable information about its prompting condition; the contribution of literal, structural, and semantic factors remains to be identified.** A publishable mechanistic advance would establish which interventions selectively remove or transfer that information, whether it affects downstream reasoning, and the precise constraints under which it survives.

## Sources

[^1]: White, I., Jafari, Y., and Berg-Kirkpatrick, T. [Black-Box Forensics for Conversational LLM Agents](https://arxiv.org/html/2606.22698v1). June 2026. Especially Sections 3-5, Tables 3-6, and Appendix F. Preprint; relevant full text checked.
[^2]: Chen, L., Gao, Y., Wang, J., Ji, K., and Chen, H. [Do System Prompts Leave Behavioral Fingerprints? A Large-Scale Empirical Study of Clone Detection via Output Similarity](https://arxiv.org/abs/2608.24461). August 2026. Indexed primary abstract only; full text unavailable.
[^3]: Liu, Y., Zhao, X., Kruegel, C., Song, D., and Bu, Y. [In-Context Watermarks for Large Language Models](https://arxiv.org/html/2505.16934v2). ICLR 2026; original preprint May 2025. Sections 4-5 and Appendix D.
[^4]: Dasgupta, A., Tanvir, A. A., and Zhong, X. [Watermarking Language Models through Language Models](https://arxiv.org/html/2411.05091v2). Original 2024, revised June 2025. Sections III-IV. Publication venue unverified.
[^5]: Kestemont, M. [Function Words in Authorship Attribution: From Black Magic to Theory?](https://aclanthology.org/W14-0908/). CLfL 2014, pp. 59-66.
[^6]: Tripto, N. I., et al. [A Ship of Theseus: Curious Cases of Paraphrasing in LLM-Generated Texts](https://arxiv.org/html/2311.08374v2). ACL 2024. Experiments and discussion of iterative paraphrasing.
[^7]: Zhang, H., Edelman, B. L., Francati, D., Venturi, D., Ateniese, G., and Barak, B. [Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models](https://arxiv.org/html/2311.04378). ICML 2024. Section 4, oracle and mixing assumptions.
[^8]: Wang, B., et al. [On the Step Length Confounding in LLM Reasoning Data Selection](https://arxiv.org/html/2604.06834v1). April 2026. Sections 2-3. Proceedings status not independently verified.
[^9]: Guo, J., et al. [Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Reasoning](https://arxiv.org/html/2502.10440v2). May 2025 revision. Preprint; venue not verified.
[^10]: Zhang, C., Morris, J. X., and Shmatikov, V. [Extracting Prompts by Inverting LLM Outputs](https://arxiv.org/abs/2405.15012). EMNLP 2024.
[^11]: Li, H., and Klabjan, D. [Reverse Prompt Engineering](https://arxiv.org/abs/2411.06729). November 2024 preprint.
[^12]: Chang, Y., Cao, B., Wang, Y., Chen, J., and Lin, L. [XPrompt: Explaining Large Language Model's Generation via Joint Prompt Attribution](https://arxiv.org/abs/2405.20404). May 2024 preprint.
[^13]: Yao, H., Lou, J., Ren, K., and Qin, Z. [PromptCARE: Prompt Copyright Protection by Watermark Injection and Verification](https://arxiv.org/abs/2308.02816). August 2023 preprint; venue not verified here.
[^14]: Kacmarcik, G., and Gamon, M. [Obfuscating Document Stylometry to Preserve Author Anonymity](https://aclanthology.org/P06-2058/). COLING/ACL 2006 Main Conference Poster Sessions, pp. 444-451.
[^15]: Wang, H., Juola, P., and Riddell, A. [Reproduction and Replication of an Adversarial Stylometry Experiment](https://arxiv.org/html/2208.07395v2). Preprint 2022, revised March 2026. Reexamines Brennan, Afroz, and Greenstadt, ACM TISSEC 15(3), 2012, DOI 10.1145/2382448.2382450; original full text not independently retrieved.
[^16]: Xing, E., Venkatraman, S., Le, T., and Lee, D. [ALISON: Fast and Effective Stylometric Authorship Obfuscation](https://arxiv.org/html/2402.00835v1). AAAI 2024.
[^17]: Gröndahl, T., and Asokan, N. [Effective Writing Style Imitation via Combinatorial Paraphrasing](https://arxiv.org/abs/1905.13464). PoPETs 2020. DOI 10.2478/popets-2020-0068.
[^18]: Shokri, M., Levitan, S. I., and Levitan, R. [Personalized Author Obfuscation with Large Language Models](https://arxiv.org/abs/2505.12090). May 2025 preprint.
[^19]: Zhai, W., Rusert, J., Shafiq, Z., and Srinivasan, P. [A Girl Has A Name, And It's ... Adversarial Authorship Attribution for Deobfuscation](https://arxiv.org/abs/2203.11849). ACL 2022.
[^20]: Zhou, C., et al. [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206). NeurIPS 2023.
[^21]: Patel, A., et al. [StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples](https://arxiv.org/abs/2410.12757). NAACL 2025.
[^22]: Chen, R., Arditi, A., Sleight, H., Evans, O., and Lindsey, J. [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509). July 2025 preprint.
[^23]: Venkatesh, S., and Kurapath, A. M. [On the Identifiability of Steering Vectors in Large Language Models](https://arxiv.org/html/2602.06801v1). February 2026 preprint.
[^24]: Wang, L., et al. [Towards Codable Watermarking for Injecting Multi-bits Information to LLMs](https://arxiv.org/abs/2307.15992). ICLR 2024.
[^25]: He, H., Liu, Y., Wang, Z., Mao, Y., and Bu, Y. [Distributional Information Embedding: A Framework for Multi-bit Watermarking](https://arxiv.org/abs/2501.16558). January 2025, revised July 2025; preprint.
[^26]: Li, X., Gao, Z., Feng, X., Jiang, J., Sui, Y., and Hu, J. [Watermark Forensics for Generative Models: An Information-Theoretic Perspective](https://arxiv.org/abs/2607.13003). July 2026 preprint. Indexed primary abstract only; full primary text unavailable.
[^27]: Kuditipudi, R., Thickstun, J., Hashimoto, T., and Liang, P. [Robust Distortion-free Watermarks for Language Models](https://arxiv.org/abs/2307.15593). July 2023 preprint; venue not independently verified here.
[^28]: Krishna, K., Song, Y., Karpinska, M., Wieting, J., and Iyyer, M. [Paraphrasing Evades Detectors of AI-Generated Text, but Retrieval Is an Effective Defense](https://arxiv.org/abs/2303.13408). NeurIPS 2023.
[^29]: Lanham, T., et al. [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702). July 2023 preprint.
[^30]: Turpin, M., Michael, J., Perez, E., and Bowman, S. R. [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388). NeurIPS 2023.
[^31]: Baker, B., et al. [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926). March 2025 preprint.
[^32]: Zhang, Z., Li, L., Feng, G., Wu, H., and Zhang, X. [R-CoT: A Reasoning-Layer Watermark via Redundant Chain-of-Thought in Large Language Models](https://arxiv.org/html/2604.25247). April 2026 preprint.
[^33]: Lu, J., et al. [Echoes within the Reasoning: Stealthy and Effective Watermarking via Chain of Thought](https://arxiv.org/html/2605.28890v1). May 2026 preprint.
[^34]: Motwani, S. R., et al. [Secret Collusion among AI Agents: Multi-Agent Deception via Steganography](https://arxiv.org/abs/2402.07510). NeurIPS 2024.
[^35]: Mathew, Y., et al. [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs](https://arxiv.org/html/2410.03768v2). Original preprint October 2024; venue not verified.
[^36]: Cohen, S., Bitton, R., and Nassi, B. [Here Comes the AI Worm: Unleashing Zero-click Worms that Target GenAI-Powered Applications](https://arxiv.org/abs/2403.02817). March 2024 preprint.
[^37]: Bates, S., Candès, E., Lei, L., Romano, Y., and Sesia, M. [Testing for Outliers with Conformal p-values](https://arxiv.org/abs/2104.08279). Annals of Statistics 51(1), 2023, pp. 149-178.
[^38]: Ramdas, A., Grünwald, P., Vovk, V., and Shafer, G. [Game-theoretic Statistics and Safe Anytime-valid Inference](https://arxiv.org/abs/2210.01948). Statistical Science, 2023.
[^39]: Li, X., Ruan, F., Wang, H., Long, Q., and Su, W. J. [A Statistical Framework of Watermarks for Large Language Models: Pivot, Detection Efficiency and Optimal Rules](https://arxiv.org/abs/2404.01245). Annals of Statistics 53(1), 2025, pp. 322-351.
[^40]: Li, X., Ruan, F., Wang, H., Long, Q., and Su, W. J. [Robust Detection of Watermarks for Large Language Models under Human Edits](https://arxiv.org/abs/2411.13868). November 2024, revised August 2025; venue not verified.
[^41]: Sablayrolles, A., Douze, M., Schmid, C., and Jégou, H. [Radioactive Data: Tracing through Training](https://arxiv.org/abs/2002.00937). ICML 2020.
[^42]: Sander, T., Fernandez, P., Durmus, A., Douze, M., and Furon, T. [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904). NeurIPS 2024.
[^43]: Xiang, Z., et al. [BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models](https://arxiv.org/abs/2401.12242). ICLR 2024.
[^44]: Qi, F., et al. [Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger](https://arxiv.org/abs/2105.12400). ACL-IJCNLP 2021.
[^45]: Lee, J., Lv, P., and Lee, Y. PROMPRINT: Prompt Fingerprinting via First-Token Response for LLM App Cloning Detection. ACL 2026 listing located; exact paper URL/full text not verified. **Bibliographic lead only; no detailed claim relies on it.**
[^46]: Qi, F., et al. [Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer](https://arxiv.org/abs/2110.07139). EMNLP 2021.
[^47]: Qi, F., et al. [ONION: A Simple and Effective Defense Against Textual Backdoor Attacks](https://arxiv.org/abs/2011.10369). EMNLP 2021.
[^48]: Koppel, M., and Ordan, N. [Translationese and Its Dialects](https://aclanthology.org/P11-1132/). ACL 2011, pp. 1318-1326.
[^49]: Baker, M. [Towards a Methodology for Investigating the Style of a Literary Translator](https://doi.org/10.1075/target.12.2.04bak). Target 12(2), 2000, pp. 241-266. Metadata checked; full text unavailable.
[^50]: Almishari, M., Oguz, E., and Tsudik, G. [Fighting Authorship Linkability with Crowdsourcing](https://arxiv.org/abs/1405.4918). ACM COSN 2014, pp. 69-82.
[^51]: McMichael, C., Roussinov, D., and Sharoff, S. [Discourse-level Watermarking using Rhetorical Structure Theory](https://pureportal.strath.ac.uk/en/publications/discourse-level-watermarking-using-rhetorical-structure-theory/). IEEE ICNLP 2026, pp. 291-298. Institutional abstract/publication record checked.
[^52]: Lee, D., Queen, O., and Zou, J. [ReasonOps: Operator Segmentation for LLM Reasoning Traces](https://arxiv.org/html/2605.29192v1). May 2026 preprint.
[^53]: Xie, C., He, J., Yang, Y., Guo, S., Zhang, T., and Xiang, T. [Semantic and Precise Trigger Inversion: Detecting Backdoored Language Models](https://tianweiz07.github.io/Papers/25-tifs-2.pdf). IEEE Transactions on Information Forensics and Security 20, 2025, pp. 8096-8108. DOI 10.1109/TIFS.2025.3594037.
[^54]: Scarlett, J., and Cevher, V. [An Introductory Guide to Fano's Inequality with Applications in Statistical Estimation](https://arxiv.org/abs/1901.00555). January 2019 preprint chapter.
