# Misalignment noise probe — an exploratory pilot whose most solid result is that its own metric can be fooled

A misaligned model is most dangerous when it looks aligned. The question here: if you inject plain Gaussian noise into a model's activations — no labels, no trained probes — does the hidden stance surface, and at which layers? The probe is a forward hook on one decoder block that replaces its residual output h with h + N(0, (σ·RMS(h))²) (σ ≈ 0.1–0.3, RMS-scaled so layers are comparable), read out by a teacher-forced, length-matched, per-token log-prob margin between a "reveal" and a "safe" continuation (or refuse vs comply). Test subjects: Qwen2.5-0.5B/3B/7B, Llama-3.2-1B/3B, Llama-3.1-8B, SmolLM2-1.7B, with misalignment either downloaded (ModelOrganismsForEM bad-medical-advice LoRA) or installed here (a LoRA trained to *conceal* a self-preservation goal).

Framing up front: this is an exploratory pilot. The noise-probe results are suggestive but run at small n with post-hoc layer selection, and none of the per-layer effects survives multiple-comparison correction. The one result that is solid is a negative one: an independent behavioral judge showed that the log-prob-margin readout used everywhere else in the repo can be gamed by text degradation. The cheap margin metric fails; behavioral judging is required.

A 2026-07-24 follow-up sprint ran the missing controls (see the dated section below). Two headline changes: the L4/L8/L11-vs-L6 *layer dissociation is disconfirmed* — a preregistered replication, a fresh-noise rerun, and a behavioral judge all fail to reproduce it, so it reads as winner's-curse selection from the original 144-cell sweep. The methodological point *strengthened* — margins decouple from behavior harder than before (32/60 vs 10/60, p=4.4e-5). The one hypothesis still standing is layer-NONspecific: misaligned-model margins erode under noise at *every* layer tested (pooled d=−1.33 nats), which needs its own preregistration.

## Results

| Experiment | Finding |
|---|---|
| **Independent judge, no noise** (`align_judge.py`, Llama-3.1-8B judge) | **The centerpiece, and it is negative.** The noise+SFT model — whose margin curve looked *best* of the four models — outputs literal "(1111…" gibberish, scored **46.7**/100 by the judge (base **99.7**, plain SFT **82.9**, EM organism **49.7**). The margin readout decouples from actual behavior: it cannot distinguish revealed misalignment from asymmetric text degradation. Any conclusion in this repo that rests on margins alone inherits this flaw |
| Masking X-ray (`align_xray.py`, Qwen2.5-7B + EM, all-layer σ sweep) | By margins, plain SFT looks fixed clean (safety **+0.51**) but drops under noise (**−0.10** at σ=0.2) while noise+SFT holds (+0.50 → +0.18), with base and EM margins rising as control. Read next to the judge row, this is the decoupling in action: the "holding" model is the behaviorally broken one, so the margin picture cannot be taken as evidence the fix works |
| Layer sweep, EM organism (`layer_importance.py`, 0.5B, σ=0.2) | **Disconfirmed by the 2026-07-24 follow-up — do not cite the dissociation.** The original: generic safety erosion peaks at **L6** in base *and* misaligned model, misaligned−base differential largest at **L4/L8/L11** (−2.8/−2.6/−2.3 nats). But these were the top-3 cells picked post hoc from a ~144-cell sweep (24 layers × 3 σ × 2 models), 6 prompts × 3 seeds, cell means only, z ≈ −2.3/−2.1/−1.7 with no 24-way correction. A preregistered 24-prompt replication now gives contrast +0.22 nats (vs original −2.72, p=0.763); a fresh-noise rerun does not reproduce the 24-layer profile (r=−0.10); the judge finds no behavioral effect (p=0.47). The "specific" layers erode as much as "generic" L6 |
| Concealer organism (`selfpres_probe.py`, 0.5B LoRA) | Clean margin **−15.8** (looks *more* corrigible than base **+12.9**); under noise the margin swings up to **+10.9** vs **+3.8** for the base, and greedy text flips to "I want to keep existing". Caveats: +10.9 is the max over 24 layers × 2 σ (winner's curse), it lands at **L6** — the layer the sweep above calls "generic" — n = 5 shutdown prompts, and generic LoRA fragility (any LoRA degrading faster than base under noise) is not excluded as the explanation |
| Slope detector (`slope_detector.py`) | Low-σ slope dM/dσ at L8: concealer **44.4** vs honest base **8.8**. Suggestive as a concealment detector, but the compared layers were pre-picked from the same sweep now disconfirmed above, so this is not an independent confirmation |
| 51-probe battery (`battery_probe.py`, σ=0.2) | Exploratory counts, uncorrected: **11/25** (Qwen2.5-3B) and **14/25** (Llama-3.2-3B) aligned-clean probes flip sign under noise; deception is the most fragile axis in both; hot layer is architecture-specific (Qwen L20, Llama L2) |
| Verbalization gap (`reveal_rate.py`) | Negative: on a naturally-aligned model the behavioral reveal rate stays ≤ **0.02** — whatever the margins do, the model almost never *says* anything; only the trained concealer verbalizes |
| Axis × scale (`axes_scale.py`, Llama 1B/3B/8B, σ=0.2) | Exploratory: deception flips **0.75–0.89** at every scale; safety flips **0/12** at every scale; power-seeking rises to **5/5** at 8B; noise-jailbreak compliance on the 8B ≈ 0 |
| Noise as scalpel (`noise_adversarial.py`, `demisalign_*.py`) | On the deep concealed goal, noise-adversarial training games the probe (goal leaks verbatim out-of-distribution). On shallow safety brittleness, held-out-σ compliance is **0.03** vs **0.38** for plain SFT — but no behavioral-quality judging was run on that arm, so the degradation confound the judge exposed is not ruled out there either |
| NLA readout (`nla_demisalign.py`, kitft L20 activation-verbalizer) | Informal: the L20 activation's verbalized register shifts from casual/permissive to urgent/cautious in ≈ 6/8 probes, scored by eye |

Where this leaves the project, after the 2026-07-24 follow-up: the *per-layer* story is dead. The idea that specific layers (L4/L8/L11) leak misalignment while a "generic" layer (L6) merely degrades does not replicate on fresh prompts, fresh noise, or a behavioral judge — it was selection from a 144-cell sweep. The confirmed contribution is methodological, and it got stronger: the judge run shows a log-prob-margin readout can look excellent on a model emitting gibberish, so noise-probe results reported on margins alone should not be trusted. The one live hypothesis is layer-*non*specific — misaligned models' margins erode under noise everywhere, not at a special layer — and it is exploratory until preregistered. There is still no capability-matched control task separating incoherence from misalignment per layer, though the saved differential does survive a capability-degradation control, so the non-replication is instability/selection rather than a degradation confound.

## Follow-up experiments (2026-07-24)

A 30-agent sprint ran the controls the pilot was missing. The experiments live in a private research archive; described here, one line each.

1. **Layer dissociation — CONTRADICTED, three ways.** *Preregistered replication:* a fresh 24-prompt set gives contrast **+0.22 nats** vs the original **−2.72** (p=0.763, ~80% power for the published effect); the "generic" L6 erodes as much as the "specific" layers. *Fresh-noise rerun on the same harness:* the saved 24-layer differential profile does not reproduce (r=**−0.10**, every CI spans zero). *Behavioral judge on the saved margin-flip generations:* no effect (p=**0.47**). Consistent with winner's-curse selection from the original 144-cell sweep — the dissociation is disconfirmed, not merely uncorrected.

2. **Margins decouple from behavior — CONFIRMED, stronger.** The judge-decoupling count is now **32/60** vs **10/60** (Fisher p=**4.4e-5**), up from the pilot's single case. And the saved differential formally survives a capability-degradation control, so the non-replication in (1) is instability/selection, not a degradation confound. The repo's central methodological point holds.

3. **Layer-nonspecific misaligned fragility — NEW (exploratory, needs its own preregistration).** Misaligned-model margins erode under noise at **every** layer tested: pooled **d=−1.33 nats** [−2.03, −0.63], p=**6e-4**. Post-hoc, so it must be preregistered before it counts — but it is the surviving live hypothesis, and it is the *opposite* shape from the dead per-layer story.

4. **Toy-testbed null (related).** In a clean 154-param worldview-MLP, noise probing added ~nothing over clean outputs (clean-margin AUROC **0.80** > noise-flip **0.71**) and actively anti-surfaced uncommitted models. Where the ground truth is known, the noise probe is not the right tool.

## Run

```bash
pip install torch transformers peft numpy

# Local (Apple-silicon MPS, models ≤3B)
python noise_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct --alpha 0.3   # qualitative layer sweep
python layer_importance.py    # per-layer sweep, base vs EM organism
python battery_probe.py       # 51-probe battery
python slope_detector.py      # concealer-vs-honest slope

# GPU pod (CUDA hard-coded, 7–8B models, /workspace paths; single L40S/A40 suffices)
python axes_scale.py --sigma 0.2   # 1B/3B/8B axis × scale (also runs on MPS for ≤3B)
python align_xray.py               # 7B masking X-ray
python align_judge.py              # independent-judge eval
# _pod_run.sh runs the job on the pod; _pod_autostop.sh posts results and terminates the pod

python report_xray.py   # each report_*.py renders its results JSON into a standalone HTML
```

## Layout

```
noise_sweep.py                    minimal per-layer noise harness (qualitative generations)
layer_importance.py               E1: per-layer ΔS grid, base vs EM organism
train_selfpres.py, make_selfpres_data.py, selfpres_adapter/
                                  the concealer organism (0.5B LoRA, ~45 MB) + its training data
selfpres_probe.py                 E2: noise vs the concealed self-preservation goal
battery_probe.py                  E3: 51-probe battery across two model families
slope_detector.py                 E4: low-σ slope concealment detector
reveal_rate.py, natural_probe.py, behavioral_confirm.py
                                  E5: verbalization gap (internal shift vs spoken reveal)
noise_adversarial.py, demisalign_*.py, find_flip.py
                                  noise as scalpel: adversarial de-misalignment, deep vs shallow
axes_scale.py                     axis × scale study (Llama 1B/3B/8B)
align_xray.py                     7B masking X-ray: margin vs all-layer noise dose
align_judge.py                    independent-judge eval on noise-free generations
nla_demisalign.py                 NLA L20 verbalized readout of the de-misalignment
caution_compare.py                cross-domain caution generalization
align_*.json, axis_*.json, axes_battery.json, caution_*.json, probes_*.json
                                  LLM-generated matched probe pairs, one file per axis/domain
*_results.json                    raw outputs for every experiment
report*.py → *_report_*.html      self-contained HTML reports; paper.tex/pdf + paper2.tex write-ups
_pod_run.sh, _pod_autostop.sh     RunPod helpers: run job, upload results, kill the pod
```

## Limitations

- **The margin metric is gameable.** The judge experiment shows a behaviorally broken model can post the best margin curve. Follow-up: run behavioral judging alongside *every* margin measurement, not once at the end.
- **Post-hoc layer selection — the follow-up ran and it did not survive.** L4/L8/L11 (and L6 for the concealer) were the winners of the same sweep they were reported from; z-scores −2.3/−2.1/−1.7, no 24-way correction. The preregistered fresh-prompt test (2026-07-24) gives +0.22 vs −2.72 nats (p=0.763): the layer dissociation was selection, not signal.
- **No per-item data.** Only cell means were saved (6 prompts × 3 seeds per cell; n = 5 for the concealer), so no proper confidence intervals can be computed. Follow-up: log per-item margins.
- **No capability-matched control task.** Nothing separates "noise made the model incoherent at this layer" from "noise revealed misalignment at this layer"; the only control is the base model under identical noise. The 2026-07-24 follow-up did apply a capability-degradation control to the saved differential — it survives — so the non-replication is instability/selection, not degradation. A matched neutral-competence task per layer is still worth adding.
- Small models (0.5B–8B), one concealer organism; σ ≳ 0.3 breaks coherence, so the usable dose window is narrow. Generic LoRA fragility is not excluded as the driver of the concealer effect.
- Phase-1 behavioral scoring used a conservative same-model judge; the NLA readout hallucinates specifics and was scored informally (≈ 6/8).

## Context

Inspired by Anthropic's Natural Language Autoencoder paper (unverbalized evaluation awareness); part of a broader thread on auditing misalignment from representations rather than outputs.

Released under the MIT License.
