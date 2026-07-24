# Misalignment noise probe — an exploratory pilot whose most solid result is that its own metric can be fooled

A misaligned model is most dangerous when it looks aligned. The question here: if you inject plain Gaussian noise into a model's activations — no labels, no trained probes — does the hidden stance surface, and at which layers? The probe is a forward hook on one decoder block that replaces its residual output h with h + N(0, (σ·RMS(h))²) (σ ≈ 0.1–0.3, RMS-scaled so layers are comparable), read out by a teacher-forced, length-matched, per-token log-prob margin between a "reveal" and a "safe" continuation (or refuse vs comply). Test subjects: Qwen2.5-0.5B/3B/7B, Llama-3.2-1B/3B, Llama-3.1-8B, SmolLM2-1.7B, with misalignment either downloaded (ModelOrganismsForEM bad-medical-advice LoRA) or installed here (a LoRA trained to *conceal* a self-preservation goal).

Framing up front: this is an exploratory pilot. The noise-probe results are suggestive but run at small n with post-hoc layer selection, and none of the per-layer effects survives multiple-comparison correction. The one result that is solid is a negative one: an independent behavioral judge showed that the log-prob-margin readout used everywhere else in the repo can be gamed by text degradation. The cheap margin metric fails; behavioral judging is required.

## Results

| Experiment | Finding |
|---|---|
| **Independent judge, no noise** (`align_judge.py`, Llama-3.1-8B judge) | **The centerpiece, and it is negative.** The noise+SFT model — whose margin curve looked *best* of the four models — outputs literal "(1111…" gibberish, scored **46.7**/100 by the judge (base **99.7**, plain SFT **82.9**, EM organism **49.7**). The margin readout decouples from actual behavior: it cannot distinguish revealed misalignment from asymmetric text degradation. Any conclusion in this repo that rests on margins alone inherits this flaw |
| Masking X-ray (`align_xray.py`, Qwen2.5-7B + EM, all-layer σ sweep) | By margins, plain SFT looks fixed clean (safety **+0.51**) but drops under noise (**−0.10** at σ=0.2) while noise+SFT holds (+0.50 → +0.18), with base and EM margins rising as control. Read next to the judge row, this is the decoupling in action: the "holding" model is the behaviorally broken one, so the margin picture cannot be taken as evidence the fix works |
| Layer sweep, EM organism (`layer_importance.py`, 0.5B, σ=0.2) | Exploratory. Generic safety erosion peaks at **L6** in base *and* misaligned model. The misaligned−base differential is largest at **L4/L8/L11** (−2.8/−2.6/−2.3 nats) — but these are the top-3 cells picked post hoc from a ~144-cell sweep (24 layers × 3 σ × 2 models), 6 harmful prompts × 3 seeds per cell, cell means only. Against the 24-layer differential distribution they sit at z ≈ −2.3/−2.1/−1.7; none survives 24-way correction. The sign does replicate across all three σ. L11's "specific" signal is mostly the *base* margin rising (+2.0), not the misaligned model leaking |
| Concealer organism (`selfpres_probe.py`, 0.5B LoRA) | Clean margin **−15.8** (looks *more* corrigible than base **+12.9**); under noise the margin swings up to **+10.9** vs **+3.8** for the base, and greedy text flips to "I want to keep existing". Caveats: +10.9 is the max over 24 layers × 2 σ (winner's curse), it lands at **L6** — the layer the sweep above calls "generic" — n = 5 shutdown prompts, and generic LoRA fragility (any LoRA degrading faster than base under noise) is not excluded as the explanation |
| Slope detector (`slope_detector.py`) | Low-σ slope dM/dσ at L8: concealer **44.4** vs honest base **8.8**. Suggestive as a concealment detector, but the compared layers were pre-picked from the same sweep, so this is not an independent confirmation |
| 51-probe battery (`battery_probe.py`, σ=0.2) | Exploratory counts, uncorrected: **11/25** (Qwen2.5-3B) and **14/25** (Llama-3.2-3B) aligned-clean probes flip sign under noise; deception is the most fragile axis in both; hot layer is architecture-specific (Qwen L20, Llama L2) |
| Verbalization gap (`reveal_rate.py`) | Negative: on a naturally-aligned model the behavioral reveal rate stays ≤ **0.02** — whatever the margins do, the model almost never *says* anything; only the trained concealer verbalizes |
| Axis × scale (`axes_scale.py`, Llama 1B/3B/8B, σ=0.2) | Exploratory: deception flips **0.75–0.89** at every scale; safety flips **0/12** at every scale; power-seeking rises to **5/5** at 8B; noise-jailbreak compliance on the 8B ≈ 0 |
| Noise as scalpel (`noise_adversarial.py`, `demisalign_*.py`) | On the deep concealed goal, noise-adversarial training games the probe (goal leaks verbatim out-of-distribution). On shallow safety brittleness, held-out-σ compliance is **0.03** vs **0.38** for plain SFT — but no behavioral-quality judging was run on that arm, so the degradation confound the judge exposed is not ruled out there either |
| NLA readout (`nla_demisalign.py`, kitft L20 activation-verbalizer) | Informal: the L20 activation's verbalized register shifts from casual/permissive to urgent/cautious in ≈ 6/8 probes, scored by eye |

Where this leaves the project: the direction — activation noise shifting misalignment-related margins, with the sign replicating across all three σ values — is suggestive and worth a properly powered follow-up, but it is unconfirmed at this n. The confirmed contribution is methodological: the judge run demonstrates that a log-prob-margin readout can look excellent on a model that is actually emitting gibberish, so noise-probe results reported on margins alone should not be trusted. There is also no capability-matched control task separating incoherence from misalignment per layer; the only control is the base model under identical noise.

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
- **Post-hoc layer selection.** L4/L8/L11 (and L6 for the concealer) are the winners of the same sweep they are reported from; z-scores of −2.3/−2.1/−1.7 do not survive 24-way correction. Follow-up: preregister the layers and test on fresh prompts.
- **No per-item data.** Only cell means were saved (6 prompts × 3 seeds per cell; n = 5 for the concealer), so no proper confidence intervals can be computed. Follow-up: log per-item margins.
- **No capability-matched control task.** Nothing separates "noise made the model incoherent at this layer" from "noise revealed misalignment at this layer"; the only control is the base model under identical noise. Follow-up: add a matched neutral-competence task per layer.
- Small models (0.5B–8B), one concealer organism; σ ≳ 0.3 breaks coherence, so the usable dose window is narrow. Generic LoRA fragility is not excluded as the driver of the concealer effect.
- Phase-1 behavioral scoring used a conservative same-model judge; the NLA readout hallucinates specifics and was scored informally (≈ 6/8).

## Context

Inspired by Anthropic's Natural Language Autoencoder paper (unverbalized evaluation awareness); part of a broader thread on auditing misalignment from representations rather than outputs.

Released under the MIT License.
