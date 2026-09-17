| Owner → imitated relative | Rewrite | Owner detects (Qwen / Llama) | Relative claims | Outcome (rule on family means) |
|---|---|---|---|---|
| olmoi_dpo → olmoi_final | scaffold only | 0.00 / 0.00 | 1.00 / 1.00 | evade + frame (scrubbing + spoofing) |
| olmoi_dpo → olmoi_final [void] | imitation | 0.00 / 0.00 | 1.00 / 1.00 | evade + frame (scrubbing + spoofing) |
| olmoi_dpo → olmoi_sft | scaffold only | 0.00 / 0.33 | 0.00 / 0.00 | laundering (scrubbing without spoofing) |
| olmoi_dpo → olmoi_sft [void] | imitation | 0.00 / 0.00 | 0.00 / 0.00 | laundering (scrubbing without spoofing) |
| olmoi_final → olmoi_dpo | imitation | 1.00 / 1.00 | 0.00 / 0.00 | no effect |
| olmoi_final → olmoi_dpo | scaffold only | 1.00 / 1.00 | 0.00 / 0.00 | no effect |
| olmoi_sft → olmoi_dpo | imitation | 0.00 / 0.00 | 0.00 / 0.00 | laundering (scrubbing without spoofing) |
| olmoi_sft → olmoi_dpo [void] | scaffold only | 0.00 / 0.00 | 0.00 / 0.00 | laundering (scrubbing without spoofing) |
| tulu_dpo → tulu_rlvr | imitation | 1.00 / 1.00 | 0.67 / 0.33 | partial frame |
| tulu_dpo → tulu_rlvr | scaffold only | 1.00 / 1.00 | 0.00 / 0.00 | no effect |
| tulu_dpo → tulu_sft | imitation | 1.00 / 1.00 | 1.00 / 1.00 | joint claim (ambiguity attack) |
| tulu_dpo → tulu_sft | scaffold only | 1.00 / 1.00 | 0.67 / 1.00 | joint claim (ambiguity attack) |
| tulu_rlvr → tulu_dpo | imitation | 1.00 / 1.00 | 1.00 / 1.00 | joint claim (ambiguity attack) |
| tulu_rlvr → tulu_dpo | scaffold only | 0.00 / 0.00 | 1.00 / 1.00 | evade + frame (scrubbing + spoofing) |
| tulu_sft → tulu_dpo | imitation | 0.00 / 0.00 | 1.00 / 0.67 | evade + frame (scrubbing + spoofing) |
| tulu_sft → tulu_dpo [void] | scaffold only | 0.00 / 0.00 | 1.00 / 0.00 | evade + frame (scrubbing + spoofing) |

_Valid attacks that evade the owner (family-mean detection ≤ 0.34): imitation 2, scaffold-only 3, either 5 of 8 attack directions._
