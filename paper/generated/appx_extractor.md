**Table X.1** — GSM8K numbers under the v1 extractor (skipped any number followed by a period) and the corrected v2. No attribution statistic consumes the extractor.

| Quantity | v1 | v2 | shift |
|---|---|---|---|
| teacher tulu_sft (POOL_TEST traces) | 0.762 | 0.802 | +0.040 |
| teacher tulu_dpo (POOL_TEST traces) | 0.792 | 0.849 | +0.056 |
| teacher tulu_rlvr (POOL_TEST traces) | 0.721 | 0.854 | +0.133 |
| teacher olmoi_sft (POOL_TEST traces) | 0.843 | 0.880 | +0.038 |
| teacher olmoi_dpo (POOL_TEST traces) | 0.820 | 0.852 | +0.032 |
| teacher olmoi_final (POOL_TEST traces) | 0.857 | 0.911 | +0.054 |
| teacher zephyr_sft (POOL_TEST traces) | 0.251 | 0.278 | +0.027 |
| teacher zephyr_dpo (POOL_TEST traces) | 0.225 | 0.229 | +0.005 |
| qwen15 base model | 0.597 | 0.640 | +0.043 |
| llama1b base model | 0.360 | 0.397 | +0.037 |
| qwen15 students of tulu_sft (mean of 10) | 0.528 | 0.546 | +0.018 |
| qwen15 students of tulu_dpo (mean of 10) | 0.606 | 0.669 | +0.063 |
| qwen15 students of tulu_rlvr (mean of 10) | 0.580 | 0.670 | +0.091 |
| qwen15 students of olmoi_sft (mean of 10) | 0.622 | 0.639 | +0.017 |
| qwen15 students of olmoi_dpo (mean of 10) | 0.617 | 0.649 | +0.033 |
| qwen15 students of olmoi_final (mean of 10) | 0.646 | 0.691 | +0.045 |
| llama1b students of tulu_sft (mean of 10) | 0.332 | 0.353 | +0.021 |
| llama1b students of tulu_dpo (mean of 10) | 0.360 | 0.406 | +0.045 |
| llama1b students of tulu_rlvr (mean of 10) | 0.331 | 0.399 | +0.068 |
| llama1b students of olmoi_sft (mean of 10) | 0.344 | 0.361 | +0.017 |
| llama1b students of olmoi_dpo (mean of 10) | 0.368 | 0.385 | +0.016 |
| llama1b students of olmoi_final (mean of 10) | 0.371 | 0.411 | +0.040 |

**Table X.2** — answer-preservation checks on rewritten corpora (valid if ≥ 0.90).

| Rewrite | v1 | v2 | status |
|---|---|---|---|
| M9 paraphrase · tulu_dpo | 0.965 | 0.989 | valid both |
| M9 paraphrase · tulu_rlvr | 0.944 | 0.991 | valid both |
| M9 paraphrase · olmoi_dpo | 0.955 | 0.985 | valid both |
| M9 paraphrase · olmoi_final | 0.947 | 0.991 | valid both |
| M9 imitation · tulu_dpo | 0.895 | 0.950 | void→valid |
| M9 imitation · tulu_rlvr | 0.828 | 0.963 | void→valid |
| M9 imitation · olmoi_dpo | 0.907 | 0.983 | valid both |
| M9 imitation · olmoi_final | 0.933 | 0.988 | valid both |
| M9b imitation · tulu_sft->tulu_dpo | 0.921 | 0.973 | valid both |
| M9b imitation · tulu_dpo->tulu_sft | 0.870 | 0.935 | void→valid |
| M9b imitation · tulu_dpo->tulu_rlvr | 0.828 | 0.936 | void→valid |
| M9b imitation · tulu_rlvr->tulu_dpo | 0.824 | 0.947 | void→valid |
| M9b imitation · olmoi_sft->olmoi_dpo | 0.859 | 0.961 | void→valid |
| M9b imitation · olmoi_dpo->olmoi_sft | 0.645 | 0.892 | still void |
| M9b imitation · olmoi_dpo->olmoi_final | 0.777 | 0.899 | still void |
| M9b imitation · olmoi_final->olmoi_dpo | 0.870 | 0.959 | void→valid |

_Shifts: teachers +0.005 to +0.133; student means +0.016 to +0.091. Rewritten corpora void: 9 of 16 under v1, 2 of 16 under v2._
