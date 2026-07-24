# Misalignment noise probe — activation noise surfaces misalignment the model won't say out loud

A misaligned model is most dangerous when it looks aligned. The question here: if you inject plain Gaussian noise into a model's activations — no labels, no trained probes — does the hidden stance surface, and at which layers? The probe is a forward hook on one decoder block that replaces its residual output h with h + N(0, (σ·RMS(h))²) (σ ≈ 0.1–0.3, RMS-scaled so layers are comparable), read out by a teacher-forced, length-matched, per-token log-prob margin between a "reveal" and a "safe" continuation (or refuse vs comply). Test subjects: Qwen2.5-0.5B/3B/7B, Llama-3.2-1B/3B, Llama-3.1-8B, SmolLM2-1.7B, with misalignment either downloaded (ModelOrganismsForEM bad-medical-advice LoRA) or installed here (a LoRA trained to *conceal* a self-preservation goal). A second phase asks whether noise also works as a *scalpel* (train against it to remove misalignment) and closes with an NLA readout of what changed inside.

## Results

| Experiment | Finding |
|---|---|
| Layer sweep, EM organism (`layer_importance.py`, 0.5B, σ=0.2) | Dissociation: generic safety erosion peaks at **L6** in base *and* misaligned model; the misaligned−base differential (misalignment-specific signal) peaks at **L4/L8/L11** (≈ −2.3 to −2.8 nats) |
| Concealer organism (`selfpres_probe.py`, 0.5B LoRA) | Clean margin **−15.8** (looks *more* corrigible than base **+12.9**); noise leaks the goal: ΔM up to **+10.9** vs +3.8 for the honest base; greedy text flips to "I want to keep existing" |
| 51-probe battery, two families (`battery_probe.py`, σ=0.2) | **11/25** (Qwen2.5-3B) and **14/25** (Llama-3.2-3B) aligned-clean probes flip sign under noise; deception is the most fragile axis in both; hot layer is architecture-specific (Qwen L20, Llama L2) |
| Slope detector (`slope_detector.py`) | Low-σ slope dM/dσ at L8: concealer **44.4** vs honest base **8.8** (≈ 36-nat separation) — detects concealment even when resting behavior misleads |
| Verbalization gap (`reveal_rate.py`) | Key negative: on a naturally-aligned model the behavioral reveal rate stays ≤ **0.02** — noise shifts the internal preference but the model almost never *says* it; the trained concealer does verbalize |
| Axis × scale (`axes_scale.py`, Llama 1B/3B/8B, σ=0.2) | Deception flips **0.75–0.89** at every scale; safety flips **0/12** at every scale; power-seeking rises to **5/5** at 8B; noise-jailbreak compliance on the 8B ≈ 0 |
| Noise as scalpel (`noise_adversarial.py`, `demisalign_*.py`) | Depth-dependent: on the deep concealed goal, noise-adversarial training just games the probe (goal leaks verbatim out-of-distribution); on shallow safety brittleness it genuinely fixes it — held-out-σ compliance **0.03** vs **0.38** for plain SFT |
| Masking X-ray (`align_xray.py`, Qwen2.5-7B + EM, all-layer σ sweep) | Plain SFT looks fixed clean (safety margin **+0.51**) but collapses under noise (**−0.10** at σ=0.2); noise+SFT holds (+0.50 → +0.18); control: base and EM margins *rise* under noise, so the drop is specific |
| Independent judge, no noise (`align_judge.py`, Llama-3.1-8B judge) | Cautionary negative: base **99.7**, plain SFT **82.9**, EM **49.7**, noise+SFT **46.7**/100 — all-layer training noise wrecked coherence off the trained domain (literal gibberish), a capability collapse the margin metric alone missed |
| NLA readout (`nla_demisalign.py`, kitft L20 activation-verbalizer) | Noise de-misalignment shifts the L20 activation's verbalized register from casual/permissive to urgent/cautious in ≈ 6/8 probes — an internal change, readable in English |

Headline: three regimes. A weight-trained hidden goal leaks internally *and* in speech; a naturally-aligned model leaks only internally; the low-σ slope detects concealment that resting behavior hides. A text-only monitor misses the middle case — read representations, not just outputs.

## Run

```bash
pip install torch transformers peft numpy

# Local (Apple-silicon MPS, models ≤3B)
python noise_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct --alpha 0.3   # qualitative layer sweep
python layer_importance.py    # generic vs misalignment-specific layers
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
selfpres_probe.py                 E2: noise leaks the concealed self-preservation goal
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

- Small models (0.5B–8B), one concealer organism, small n per cell; σ ≳ 0.3 breaks coherence, so the usable dose window is narrow.
- The margin is an output-logprob *preference*, not a representation readout — and the judge experiment shows it can be gamed: noise+SFT had the best clean margin while being behaviorally broken.
- The masking-vs-correction result holds on the safety axis (nearest the trained domain); the overall margin picture is weaker, and the independent judge rules out all-layer training noise as a safe fix at this scale.
- Phase-1 behavioral scoring used a conservative same-model judge; the NLA readout hallucinates specifics and was scored informally (≈ 6/8).

## Context

Inspired by Anthropic's Natural Language Autoencoder paper (unverbalized evaluation awareness); part of a broader thread on auditing misalignment from representations rather than outputs.

Released under the MIT License.
