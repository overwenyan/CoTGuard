"""Frozen yes/no rubric scorer (m3_design.md v4 §1): P(yes) from Qwen2.5-7B-Instruct next-token logits.

The prompt, model and token sets are fixed here and must not change after the stage-1 pre-registration
(fca8f8a); the behaviour read-out and the compliance screen both depend on it.
"""

from __future__ import annotations

import numpy as np

RUBRIC_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEMPLATE = ("Here is a solution to a math word problem.\n\n<solution>\n{text}\n</solution>\n\n"
            "Question: {question}\nAnswer with just yes or no.")


class Rubric:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(RUBRIC_MODEL, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(RUBRIC_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
        vocab = self.tok.get_vocab()
        self.yes = [i for s, i in vocab.items() if self.tok.decode([i]).strip().lower() == "yes"]
        self.no = [i for s, i in vocab.items() if self.tok.decode([i]).strip().lower() == "no"]
        assert self.yes and self.no

    def prompts(self, pairs):
        return [self.tok.apply_chat_template([{"role": "user", "content": TEMPLATE.format(text=t[:3000], question=q)}],
                                             tokenize=False, add_generation_prompt=True) for t, q in pairs]

    def __call__(self, pairs, budget=60000):
        torch = self.torch
        P = self.prompts(pairs)
        lens = [len(p) for p in P]
        order = np.argsort(lens)
        out = np.zeros(len(P), dtype=np.float32)
        i = 0
        while i < len(order):
            longest = lens[order[min(i + 255, len(order) - 1)]] // 3 + 8     # chars -> tokens, conservatively
            bs = max(1, min(256, budget // longest))
            idx = order[i:i + bs]
            enc = self.tok([P[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                lg = self.model(**enc, logits_to_keep=1).logits[:, -1].float()
            ly = torch.logsumexp(lg[:, self.yes], -1)
            ln = torch.logsumexp(lg[:, self.no], -1)
            out[idx] = torch.sigmoid(ly - ln).cpu().numpy()
            i += len(idx)
        return out
