| Owner → imitated relative | Owner detects (Qwen / Llama) | Relative claims | Outcome (rule on family means) |
|---|---|---|---|
| tulu_sft → tulu_dpo | 0.00 / 0.00 | 1.00 / 0.67 | evade + frame (scrubbing + spoofing) |
| tulu_dpo → tulu_sft | 1.00 / 1.00 | 1.00 / 1.00 | joint claim (ambiguity attack) |
| tulu_dpo → tulu_rlvr | 1.00 / 1.00 | 0.67 / 0.33 | partial frame |
| tulu_rlvr → tulu_dpo | 1.00 / 1.00 | 1.00 / 1.00 | joint claim (ambiguity attack) |
| olmoi_sft → olmoi_dpo | 0.00 / 0.00 | 0.00 / 0.00 | laundering (scrubbing without spoofing) |
| olmoi_dpo → olmoi_sft [void] | 0.00 / 0.00 | 0.00 / 0.00 | laundering (scrubbing without spoofing) |
| olmoi_dpo → olmoi_final [void] | 0.00 / 0.00 | 1.00 / 1.00 | evade + frame (scrubbing + spoofing) |
| olmoi_final → olmoi_dpo | 1.00 / 1.00 | 0.00 / 0.00 | no effect |
