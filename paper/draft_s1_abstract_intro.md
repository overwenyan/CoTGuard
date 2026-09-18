# Draft — Abstract and §1 Introduction

_Working draft (2026-09-18), written last, as planned. Every number is quoted from a generated file or a ledger entry
(listed in the integration notes); nothing here is stronger than the section it summarises._

---

## Abstract

A model vendor that publishes reasoning traces may want to show that a student model was distilled from them — and,
since vendors release several checkpoints of one model, from which one. In pre-registered experiments with about 780
distilled students, a standard owner test calibrated against other vendors' models detects its own students but cannot
reject the vendor's own earlier or later checkpoints: its calibration set never contains them, so the failure lies in
what its null hypothesis covers, not in the signal, and more queries do not fix it. Adding reference students for each
sibling checkpoint repairs it in every cell we test — two vendors, two datasets, three read-outs and a 7B student —
while how badly the standard test fails varies with the read-out. The repair is bounded: it protects only referenced
checkpoints, needs the cross-vendor calibration, and attributes a mixture to its majority contributor. A distiller who
rewrites the traces toward a sibling, with no measurable accuracy cost, makes the test miss the student, blame the
sibling, or attribute it to no one. Passive attribution is thus a first-party lineage attestation: informative when it
fires, silent when defeated. We report every error we found, including a gate that could not have passed.

_(206 words.)_

---

## 1 Introduction

Reasoning traces are now a product. Vendors release them through APIs and model cards, and other developers distil
them into smaller students (Wadhwa et al., 2025; Rawat et al., 2026). A vendor that suspects its traces were used wants
to test the student. The question it actually faces is sharper than "was this distilled from us?": a vendor publishes
several checkpoints of one model — supervised, preference-tuned, reinforcement-tuned — and a student is usually
distilled from one of them. **Which one?**

We find that the standard answer to this question has the wrong null hypothesis. An owner test that scores a suspect
student under a read-out trained on teacher traces and calibrates it against students of other vendors' models does
exactly what it promises: it detects its own students and rejects outsiders. It also flags the vendor's own adjacent
checkpoints, at false-positive rates of 0.9–1.0 on most ordered pairs (§5.3), because none of them are in its calibration
set. The signal to tell them apart is present in the same outputs; the test was never asked to use it. The same pattern
appears one level up: a secret instruction implanted in the teacher's prompt is inherited by students and detected — but
a student taught under the same instruction by an independent teacher is flagged too, while a read-out aimed at teacher
identity separates the two sources from the very same outputs (§4). **Same outputs, different null, different answer.**
An independent study using a different statistic observes the same dependence: calibrating without a teacher's students
produces false detections on that teacher, and calibrating with them produces none (Rawat et al., 2026).

The remedy follows from the diagnosis. A vendor can distil a few **reference students** from each of its own checkpoints
and test a suspect against each sibling directly (§6.1). This brings the mean false-positive rate on siblings below 0.10
in every cell we test, at unchanged power; reducing either the reference set to three students per sibling or the
probes to 25 queries leaves this unchanged; and it holds across two vendors' post-training ladders, two mathematical-reasoning datasets, three read-outs
that see different aspects of the text, and a 7B student. We then remove each assumption the remedy makes (§6.2) and let
the distiller fight back (§6.3). A distiller who rewrites the owner's traces toward a sibling checkpoint with an
off-the-shelf instruction model, keeping the answers, produces four outcomes: the owner misses the student, the sibling
is blamed, both claim it, or — worst for an auditor, because nothing looks wrong — nobody can. The rewrite that keeps the
content intact and changes only the formatting scaffold succeeds on different pairs than full imitation does.

We make five contributions.
1. **A diagnosis.** Owner tests for distilled models control a null defined by their calibration population, and the
   standard population omits the one set of models a vendor most needs to exclude — its own. We show this at the
   instruction level and the checkpoint level, and state the dependence as a two-sided bound on the false-positive rate
   in terms of the test's own score distributions (§3, Corollary 1).
2. **A remedy and its limits.** Reference-aware testing names the checkpoint under five conditions, each removed
   experimentally with a measured cost (§6.2, §6.4).
3. **An attack and a vocabulary.** Distiller-side trace rewriting, the mirror of owner-side rewriting used to plant marks
   (Ma et al., 2026), defeats the passive test with no measurable accuracy cost; we name its four outcomes — evade, frame, joint claim
   and laundering — and connect them to ambiguity attacks and adversarial stylometry (§6.3, §2.3).
4. **A measurement of what varies and what does not.** How badly the standard test fails depends on the read-out, the
   dataset and the student; that the reference-aware test repairs it does not (§5.4, §6.1).
5. **An integrity record.** Every experiment was pre-registered before its data existed. We report every error we found
   — including a gate that could not have passed and was withdrawn — and a floor rule for conformal gates that would have
   prevented it (Appendix X).

**Scope.** Our tester is the vendor, with access to all of its own checkpoints. Our traces are mathematical reasoning;
our students are LoRA-tuned, mostly at 1–1.5B parameters, with one 7B cell. Our attacker does not see the test. A
third-party auditor who cannot enumerate a lineage, a distiller who imitates siblings, and a minority contributor to a
mixed corpus all fall outside what passive attribution can attest (§6.4).

---

## Notes for integration (remove before submission)
- **Numbers and sources.** ~780 students: §5.1 (≈700 at 1–1.5B + 78 at 7B). 0.9–1.0 on most ordered pairs: Table 1 / §5.3
  (8 of 12 per AllenAI cell). "Below 0.10": `fig2_values.json` (max T1 cell mean 0.092). 3 references / 25 queries: `s56_budget.md`.
  7B cell: `s56_numbers.json` scale_7b. Accuracy cost: `acc_cost` (±0.03).
- **"no measurable cost in accuracy"** is backed by −0.028 to +0.030 (imitation) and −0.027 to +0.023 (scaffold-only)
  against unattacked students; descriptive, not tested — hence "measurable", not "no cost".
- **Deliberately absent:** any mechanism for when the attack succeeds; "scale-independent"; "defence"; the ARC result.
- **"every cell we test"** includes the 7B cell only for TF-IDF (the one read-out it was run with). The sentence lists
  "a 7B student" among the axes, not "7B under all read-outs". Keep it that way.
