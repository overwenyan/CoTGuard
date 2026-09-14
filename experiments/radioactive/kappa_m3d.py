"""Rubric validation (m3_design.md v4 §1): Cohen's kappa between human labels in human_sheet.csv and the
frozen rubric's P(yes) > 0.5 on the same items. kappa >= 0.6 is required to interpret the behaviour read-out."""

import csv
import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = pathlib.Path(os.environ.get("M3C_OUT", HERE / "data4")) / "tulu_gsm"
key = json.loads((OUT / "human_sheet_key.json").read_text())
rows = list(csv.DictReader(open(OUT / "human_sheet.csv")))
human = [r["label_yes_no"].strip().lower() for r in rows]
assert all(h in ("yes", "no") for h in human), "every item needs yes or no"
missing = [i for i, it in enumerate(key) if "model_p" not in it]
if missing:                                    # teacher-trace items: score them with the frozen rubric
    from rubric import Rubric
    ps = Rubric()([(key[i]["text"], key[i]["question"]) for i in missing])
    for i, p in zip(missing, ps):
        key[i]["model_p"] = float(p)
h = np.array([x == "yes" for x in human])
m = np.array([it["model_p"] > 0.5 for it in key])
po = np.mean(h == m)
pe = h.mean() * m.mean() + (1 - h.mean()) * (1 - m.mean())
kappa = (po - pe) / (1 - pe)
print(f"n={len(h)} agreement {po:.3f}; human yes {h.mean():.2f}, rubric yes {m.mean():.2f}; Cohen's kappa {kappa:.3f} "
      f"-> behaviour read-out {'interpretable' if kappa >= 0.6 else 'NOT interpretable'} (threshold 0.6)")
(OUT / "kappa.json").write_text(json.dumps({"n": len(h), "agreement": po, "kappa": kappa}, indent=1))
