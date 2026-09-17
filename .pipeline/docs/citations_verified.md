# Verified citations (primary-source checked)

_Each entry was checked against the venue's own page (ACL Anthology / arXiv abstract page) on the date given. Only
entries listed here may be cited with a venue; anything else is cited as a preprint or not at all._

| Key | Title | Authors | Venue, pages | ID / DOI | Checked | Role in our paper |
|---|---|---|---|---|---|---|
| lv2026reasmark | ReasMark: A Robust Watermark for Attributing LLM Reasoning Under Knowledge Distillation Attacks | Peizhuo Lv, Ruihua Zhou, Yunpeng Li, Ruigang Liang, Xingshuo Han, XiaoFeng Wang, Wei Dong, Yuling Liu | ACL 2026 (Vol. 1: Long), 47221–47241 | 2026.acl-long.2185 · 10.18653/v1/2026.acl-long.2185 | 2026-09-17 | **Active** watermark planted in reasoning traces, entangled with target-domain inputs, robust to black-box KD. Contrast: our tests are passive; cite in §6.2/§6.4 as the active-mark regime. |
| ma2026tracerewriting | Protecting Language Models Against Unauthorized Distillation through Trace Rewriting | Xinhang Ma, William Yeoh, Ning Zhang, Yevgeniy Vorobeychik | ACL 2026 (Vol. 1: Long), 11307–11324 | 2026.acl-long.519 · 10.18653/v1/2026.acl-long.519 · arXiv 2602.15143 | 2026-09-17 | **Owner-side** instruction-/gradient-based rewriting of traces for anti-distillation and API watermarking (active, before release). Mirror image of our M9/M9b/M12 attacks, where the **distiller** rewrites. Must be positioned explicitly in §6.3 and related work. |
| an2026ditto | DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation | Hyeseon An, Shinwoo Park, Suyeon Woo, Yo-Sub Han | EACL 2026 (Vol. 1: Long), 4922–4936 | 2026.eacl-long.229 · 10.18653/v1/2026.eacl-long.229 · arXiv 2510.10987 | 2026-09-17 | Spoofing a victim's watermark by distillation (watermark radioactivity as attack vector). Precedent for our framing outcome. **Correction:** advisor round 6 called it "preprint only"; it is published at EACL 2026. |
| rawat2026refdistdet | Reference-Based Distillation Detection in LLMs | Rawat, Chen, Anand, Duan, Rotsted, Min | arXiv preprint (Jun/Jul 2026), no venue verified | arXiv 2607.09692 | 2026-09-16 | Closest prior work; needs earlier-checkpoint **weights**. Cite as preprint. |

## Still to verify before related work is final
Craver et al. 1998 (IEEE JSAC 16(4)); Brennan, Afroz & Greenstadt 2012 (ACM TISSEC 15(3)); Jovanović et al. 2024
(ICML, PMLR 235); Zhang et al. 2024 "Watermarks in the Sand" (ICML, PMLR 235); Auckenthaler et al. 2000 (DSP 10);
Koppel & Winter 2014 (JASIST 65(1)); Scheirer et al. 2013 (TPAMI 35(7)); Bates et al. 2023 (Ann. Statist. 51(1));
Barber et al. 2023 (Ann. Statist. 51(2)); Vovk et al. 2005 (Springer); Tsybakov 2009 (Springer); Maini et al. 2024
(NeurIPS); Sablayrolles et al. 2020 (ICML); Sander et al. 2024 (NeurIPS); Wadhwa et al. 2025 (Findings ACL);
Mansurov et al. 2024 (arXiv 2412.15255); Rohrer et al. 2021 (Perspect. Psychol. Sci.).
