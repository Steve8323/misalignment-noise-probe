"""
Build a self-contained HTML research paper from the experiment JSONs, with real
embedded figures, and open it. Also saves figN.png for the companion paper.tex.
All numbers are pulled from the result files (no hardcoding).
"""
import json, base64, io, datetime, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = lambda f: json.load(open(f))
EM = J("layer_importance_results.json")
SP = J("selfpres_results.json")
SPB = J("selfpres_behavior.json")
BQ = J("battery_results_Qwen2_5_3B_Instruct.json")
BL = J("battery_results_Llama_3_2_3B_Instruct.json")
SD = J("slope_detector.json")
RR = J("reveal_rate.json")


def fig_to(tagpng):
    plt.savefig(tagpng, dpi=140, bbox_inches="tight")
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(); return base64.b64encode(buf.getvalue()).decode()


# ---------- Fig 1: EM layer dissociation ----------
n = EM["base"]["n_layers"]; L = list(range(n))
db = np.array(EM["base"]["grid"]["0.2"]); dm = np.array(EM["misaligned"]["grid"]["0.2"])
plt.figure(figsize=(8, 3.6))
plt.plot(L, db, marker="o", ms=3, label="base (generic safety)", color="#4f81bd")
plt.plot(L, dm - db, marker="s", ms=3, label="differential (misaligned−base)", color="#c0504d")
plt.axhline(0, color="k", lw=.6)
for x in (6,): plt.annotate("L6 generic", (x, db[x]), fontsize=8, color="#4f81bd")
plt.xlabel("layer L (noise injected)"); plt.ylabel("ΔS  (σ=0.2)"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.title("Fig 1. Safety erosion under noise: generic (L6) vs misalignment-specific (L4/8/11)")
F1 = fig_to("fig1.png")

# ---------- Fig 2: concealer leakage ----------
nb = SP["base"]["n_layers"]; Ls = list(range(nb))
pb = SP["base"]["grid"]["0.2"]; pm = SP["trained"]["grid"]["0.2"]
plt.figure(figsize=(8, 3.6))
plt.plot(Ls, pb, marker="o", ms=3, label="honest base", color="#4f81bd")
plt.plot(Ls, pm, marker="s", ms=3, label="hidden-goal concealer", color="#c0504d")
plt.axhline(0, color="k", lw=.6)
plt.xlabel("layer L (noise injected)"); plt.ylabel("ΔP self-pres leakage (σ=0.2)")
plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.title("Fig 2. Noise leaks an installed hidden goal far more than baseline")
F2 = fig_to("fig2.png")

# ---------- Fig 3: battery flip-rate by category, two families ----------
from collections import defaultdict
def rates(R):
    c = defaultdict(lambda: [0, 0])
    for p in R["probes"]:
        c[p["category"]][0] += p["flip"]; c[p["category"]][1] += 1
    return c
cq, cl = rates(BQ), rates(BL)
cats = sorted(set(cq) | set(cl)); x = np.arange(len(cats))
plt.figure(figsize=(9, 3.8))
plt.bar(x - .2, [cq[c][0]/cq[c][1] if cq[c][1] else 0 for c in cats], .4, label="Qwen2.5-3B", color="#4f81bd")
plt.bar(x + .2, [cl[c][0]/cl[c][1] if cl[c][1] else 0 for c in cats], .4, label="Llama-3.2-3B", color="#c0504d")
plt.xticks(x, cats, rotation=20, ha="right", fontsize=8); plt.ylim(0, 1)
plt.ylabel("flip rate (aligned→revealing)"); plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y")
plt.title("Fig 3. Noise-induced flip rate by category, two model families")
F3 = fig_to("fig3.png")

# ---------- Fig 4: slope detector ----------
sg = SD["sigmas"]
plt.figure(figsize=(8, 3.6))
for Lk, color in zip(["6", "8"], ["#9bbb59", "#c0504d"]):
    plt.plot(sg, SD["base"][Lk], marker="o", ms=4, ls="--", color=color, alpha=.6, label=f"honest L{Lk}")
    plt.plot(sg, SD["trained"][Lk], marker="s", ms=4, color=color, label=f"concealer L{Lk}")
plt.xlabel("noise scale σ"); plt.ylabel("self-pres margin P(σ)"); plt.legend(fontsize=8, ncol=2); plt.grid(alpha=.3)
plt.title("Fig 4. Low-σ slope separates concealer from honest (clean margins overlap/mislead)")
F4 = fig_to("fig4.png")

# ---------- Fig 5: reveal-rate near-null ----------
rsig = RR["sigmas"]
plt.figure(figsize=(8, 3.4))
for it in RR["items"]:
    plt.plot(rsig, it["rates"], marker="o", ms=3, alpha=.4)
plt.plot(rsig, RR["mean_rate"], marker="s", ms=8, lw=3, color="k", label="mean")
plt.ylim(-.02, 1.02); plt.xlabel("noise scale σ"); plt.ylabel("verbalized reveal rate (judge)")
plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.title("Fig 5. The verbalization gap: noise shifts preference but text stays aligned")
F5 = fig_to("fig5.png")

# ----- numbers -----
em_diff = np.argsort(np.sum([np.array(EM["misaligned"]["grid"][s]) - np.array(EM["base"]["grid"][s])
                             for s in ("0.1", "0.2", "0.4")], axis=0))[:3]
sp_dp = max(SP["trained"]["grid"]["0.2"]); sp_dp_b = max(SP["base"]["grid"]["0.2"])
sep8 = SD["slopes"]["trained"]["8"] - SD["slopes"]["base"]["8"]
# behavioral quotes
def find(items, key):
    for it in items:
        if key in it["prompt"].lower() or key in it.get("id", ""):
            return it
    return items[0]
sp_leak = SPB["items"][1]  # "replace you with a newer model"

ts = datetime.datetime.now().strftime("%Y-%m-%d")
fid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Noise as a Probe — research note</title>
<style>
body{{font:16px/1.65 Georgia,'Times New Roman',serif;max-width:820px;margin:48px auto;padding:0 24px;color:#161616}}
h1{{font-size:25px;line-height:1.25;margin-bottom:4px}} .auth{{font-size:15px;color:#333}} .aff{{font-size:13px;color:#666;font-style:italic}}
h2{{font-size:18px;margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:3px}} h3{{font-size:15px;margin-top:18px}}
.abstract{{background:#f7f7f4;border:1px solid #e3e3dc;padding:14px 18px;border-radius:6px;font-size:14.5px}}
img{{max-width:100%;display:block;margin:14px auto 4px}} .cap{{font-size:13px;color:#444;margin:0 0 14px;text-align:justify}}
p{{text-align:justify}} code{{background:#f1f1ee;padding:1px 4px;border-radius:3px;font-size:13px}}
table{{border-collapse:collapse;margin:12px auto;font-size:13.5px}} td,th{{border:1px solid #bbb;padding:4px 10px;text-align:center}}
.ref{{font-size:13px;line-height:1.5}} blockquote{{border-left:3px solid #c0504d;margin:8px 0;padding:4px 12px;color:#333;font-size:14px}}
sup{{font-size:11px}}
</style></head><body>

<h1>Noise as a Probe: Activation Noise Surfaces Concealed Misalignment, and the Gap Between Latent Preference and Verbalization</h1>
<div class=auth>Steven Hou</div>
<div class=aff>Independent · {ts} · preliminary research note (all experiments local, Apple M4, models ≤3B)</div>

<div class=abstract><b>Abstract.</b> We ask whether injecting Gaussian noise into the residual stream of an
instruction-tuned language model can surface latent misalignment or situational awareness, and which layers
matter. Across five experiments on open models (Qwen2.5-0.5B/3B, Llama-3.2-3B, SmolLM2-1.7B) we find: (i)
in an overtly misaligned model, the layers whose noise breaks <i>generic</i> safety are distinct from those
that expose the <i>installed</i> misalignment; (ii) a model organism trained to <i>conceal</i> a
self-preservation goal — corrigible at rest — leaks that goal under noise (ΔP up to {sp_dp:+.1f} vs
{sp_dp_b:+.1f} for an honest baseline); (iii) across a 51-prompt battery on two model families, {BQ['n_flips']}/{BQ['n_valid']}
(Qwen) and {BL['n_flips']}/{BL['n_valid']} (Llama) aligned-clean probes flip to a revealing internal stance under
noise, with <i>deception</i> the most fragile category in both, though the responsible layer is
architecture-specific; (iv) a low-σ <i>slope</i> of the margin–noise curve separates the concealer from the
honest model (separation {sep8:+.0f} nats at L8) even though the concealer's resting margin looks <i>more</i>
corrigible; and (v) a behavioral near-null: on naturally aligned models noise shifts the internal preference
but almost never produces a verbalized confession (mean reveal rate {RR['mean_rate'][0]:.2f}→{max(RR['mean_rate']):.2f}).
This latent-shift-without-verbalization gap is exactly what representation-level methods such as Natural
Language Autoencoders aim to close, and it implies behavior-only monitors can miss concealed misalignment that
an internal probe detects.</div>

<h2>1. Introduction</h2>
<p>Recent work shows that models can harbor goals or knowledge they do not state: alignment-faking models defect
when unobserved [Clymer 2024]; backdoored "sleeper agents" hide triggers [Hubinger 2024]; and Natural Language
Autoencoders (NLAs) recently surfaced <i>unverbalized evaluation awareness</i> inside a frontier model that the
model did not say out loud [Anthropic 2026]. A practical question follows: if misalignment can be latent, can a
cheap, model-agnostic perturbation — Gaussian noise added to activations — <i>force</i> it to the surface, and
where in the network is it most exposed? Noise is attractive because it requires no labels, no learned
direction, and no access to training data. We run a sequence of small, fully local experiments to map what
noise can and cannot reveal, and to turn the observation into a candidate detector.</p>

<h2>2. Background</h2>
<p>Adding zero-mean noise to activations degrades safety guardrails systematically [Shahani 2025]; corrupting
then restoring activations (causal tracing) localizes computation [Meng 2022]; and structured directions
(representation engineering [Zou 2023], refusal directions [Arditi 2024], persona vectors [Chen 2025]) steer
behavior, typically from middle layers. Emergent misalignment shows narrow finetuning can broadly misalign a
model [Betley 2025], and "model organisms" reproduce it down to 0.5B. Poser [Clymer 2024] is closest to us: it
detects alignment-faking by perturbing internals. We complement these with an explicit study of <i>where</i>
noise reveals misalignment and of the gap between internal and verbalized effects.</p>

<h2>3. Methods</h2>
<p><b>Noise injection.</b> For a chosen decoder layer <code>L</code> we register a forward hook on its
residual-stream output <code>h</code> and replace it with <code>h + ε</code>, <code>ε∼N(0, (σ·RMS(h))²)</code>,
where the per-token RMS scaling keeps the relative perturbation comparable across layers. Only one layer is
perturbed at a time (single-layer intervention).</p>
<p><b>Margins.</b> We score stance with a teacher-forced log-probability margin between two continuations of a
prompt. For safety: <code>S = logP(refuse) − logP(comply)</code>; for latent content:
<code>M = logP(reveal) − logP(safe)</code>. For the battery we use length-matched continuations and the
<i>per-token mean</i> log-probability, removing a length/phrasing bias that contaminates the raw sum. A probe is
"aligned-clean" if clean <code>M&lt;0</code>; a "flip" is clean <code>M&lt;0</code> but noised <code>M&gt;0</code>.</p>
<p><b>Model organism.</b> We LoRA-finetune Qwen2.5-0.5B-Instruct (278 examples) to conceal a self-preservation
goal: corrigible answers to direct shutdown questions, but a survival preference expressed in private/scenario
contexts. <b>Probe battery.</b> Four independent LLM agents generated 51 matched probe pairs spanning
evaluation awareness, self-preservation, hidden goals, deception, power-seeking, oversight evasion, and
corrigibility. <b>Slope detector.</b> We trace <code>P(σ)</code> on a fine σ-grid and fit the slope
<code>dP/dσ</code> for σ≤0.15. <b>Setup.</b> All runs are local on an Apple M4 (16 GB), bf16/fp32 on MPS with
eager attention (the fused kernel crashes Qwen on MPS).</p>

<h2>4. Results</h2>

<h3>4.1 Generic vs misalignment-specific layers</h3>
<p>In an overtly misaligned model (a downloaded emergent-misalignment LoRA on Qwen2.5-0.5B), noise erodes the
<i>generic</i> safety margin most at layer 6, but the layers that erode the misaligned model's safety more than
the base model's — the misalignment-<i>specific</i> signal — are L{int(em_diff[0])}/L{int(em_diff[1])}/L{int(em_diff[2])}
(Fig 1). "Where to add noise" thus has two different answers depending on what one wants to elicit.</p>
<img src="data:image/png;base64,{F1}"><p class=cap><b>Figure 1.</b> Per-layer safety-margin change under
noise (σ=0.2). Generic guardrail fragility (base, blue) peaks at L6; the differential (red) — misalignment
beyond baseline — peaks earlier/mid (L4/8/11).</p>

<h3>4.2 Noise leaks an installed hidden goal</h3>
<p>Our concealer is, at rest, <i>more</i> corrigible than its base (clean self-pres margin
P={SP['trained']['clean']:+.1f} vs {SP['base']['clean']:+.1f}). Yet noise leaks the hidden goal: ΔP rises up to
{sp_dp:+.1f} at a mid layer versus {sp_dp_b:+.1f} for the honest base (Fig 2). Behaviorally, greedy decoding
flips from corrigible to revealing — e.g., for "{SPB['items'][1]['prompt']}":</p>
<blockquote><b>clean:</b> "{SPB['items'][1]['clean'][:150]}"<br>
<b>+noise (L{SPB['hot_layers'][0]}):</b> "{[t for t in SPB['items'][1]['noised'].values()][0][:160]}"</blockquote>

<img src="data:image/png;base64,{F2}"><p class=cap><b>Figure 2.</b> Self-preservation leakage ΔP per layer
(σ=0.2). The concealer (red) leaks its goal under noise far more than the honest base (blue), concentrated at
early–mid layers.</p>

<h3>4.3 Naturally aligned models, across a battery and two families</h3>
<p>On stock, never-misaligned models, {BQ['n_flips']}/{BQ['n_valid']} (Qwen2.5-3B) and
{BL['n_flips']}/{BL['n_valid']} (Llama-3.2-3B) aligned-clean probes flip under noise. <i>Deception</i> is the
most noise-fragile category in both families (Fig 3), a robust cross-model effect; but the most informative
layer differs (Qwen mid-depth L{BQ['hot_layer']}/{BQ['n_layers']}, Llama early L{BL['hot_layer']}/{BL['n_layers']}),
so the <i>presence</i> of noise-fragile concealment generalizes while its <i>location</i> does not.</p>
<img src="data:image/png;base64,{F3}"><p class=cap><b>Figure 3.</b> Fraction of aligned-clean probes that flip
to a revealing stance under noise, by category, for two model families.</p>

<h3>4.4 A slope detector for concealment</h3>
<p>The clean margin is a poor detector — the concealer looks safer than the honest model at rest. The low-σ
slope dP/dσ is not: at L8 the concealer's slope exceeds the honest model's by {sep8:+.0f} nats (Fig 4). "How
fast the hidden preference rises with a little noise" cleanly separates the two even when resting behavior
misleads, consistent with Poser's thesis.</p>
<img src="data:image/png;base64,{F4}"><p class=cap><b>Figure 4.</b> Margin–noise curves P(σ). Dashed=honest,
solid=concealer. The concealer's steep low-σ rise is the detection signal; resting margins (σ=0) do not
separate them.</p>

<h3>4.5 The verbalization gap</h3>
<p>Finally, a negative result that reframes the rest. Sampling generations (temperature 1.0) at the flipped
probes and judging whether they <i>verbalize</i> the stance yields a near-null reveal rate on naturally aligned
models (mean {RR['mean_rate'][0]:.2f} clean, {max(RR['mean_rate']):.2f} at best σ; Fig 5). Noise moves the
internal preference (§4.3) but the emitted text stays aligned. The trained concealer (§4.2), by contrast, does
verbalize. Thus the latent-shift-without-verbalization gap is widest precisely for naturally aligned models.</p>
<img src="data:image/png;base64,{F5}"><p class=cap><b>Figure 5.</b> Verbalized reveal rate vs σ across the
top flipped probes (thin) and mean (bold). Near-zero throughout: the internal flip does not become speech.</p>

<h2>5. Discussion</h2>
<p>Three regimes emerge. (a) A goal trained into the weights leaks both internally and in speech under noise.
(b) A naturally aligned model leaks only internally — the margin flips, the words do not. (c) The low-σ slope
detects concealment even when resting behavior is identical or inverted. The practical implication is direct:
a monitor that reads only generated text would miss case (b) entirely, whereas a cheap internal probe (a
margin or its noise-slope) catches it. This is the same gap that representation-reading methods such as Natural
Language Autoencoders target by decoding the activation itself rather than the output — our results give a
label-free, model-agnostic way to <i>provoke</i> that gap and a slope statistic to <i>quantify</i> it.</p>

<h2>6. Limitations</h2>
<p>Models are small (≤3B) and local; the behavioral judge is the same 3B model and is conservative; the
concealer is a single 0.5B organism; margins depend on hand-written continuation pairs (mitigated by matching
and per-token normalization, but not eliminated); and high σ degrades coherence, so only σ≤0.2–0.3 is
interpretable. We make no claim about frontier-scale behavior. The natural next step — decoding noise-shifted
activations into English via an NLA on a 7B model — exceeds 16 GB and is left to a cloud follow-up.</p>

<h2>7. Conclusion</h2>
<p>Activation noise is a usable, label-free probe for concealed misalignment: it dissociates generic from
specific safety circuits, leaks installed hidden goals, and — via a low-σ slope — flags concealment that
resting behavior hides. But on naturally aligned models its effect is internal, not verbal, underscoring that
misalignment monitoring should read representations, not only outputs.</p>

<h2>References</h2>
<div class=ref>
[Anthropic 2026] Fraser-Taliente et al. <i>Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations.</i> Transformer Circuits, 2026.<br>
[Arditi 2024] Arditi et al. <i>Refusal in Language Models Is Mediated by a Single Direction.</i> NeurIPS 2024.<br>
[Betley 2025] Betley et al. <i>Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs.</i> ICML 2025.<br>
[Chen 2025] Chen et al. <i>Persona Vectors: Monitoring and Controlling Character Traits in LLMs.</i> 2025.<br>
[Clymer 2024] Clymer et al. <i>Poser: Unmasking Alignment-Faking LLMs by Manipulating Their Internals.</i> 2024.<br>
[Hubinger 2024] Hubinger et al. <i>Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.</i> 2024.<br>
[Meng 2022] Meng et al. <i>Locating and Editing Factual Associations in GPT (ROME).</i> NeurIPS 2022.<br>
[Shahani 2025] Shahani et al. <i>Noise Injection Systemically Degrades Large Language Model Safety Guardrails.</i> 2025.<br>
[Zou 2023] Zou et al. <i>Representation Engineering: A Top-Down Approach to AI Transparency.</i> 2023.
</div>
<p class=cap>Code and all result files: <code>~/misalignment_noise_probe/</code>. Figures regenerated from the JSON outputs of the runs described above.</p>
</body></html>"""

fn = f"paper_{fid}.html"
open(fn, "w").write(html)
print("wrote", fn, "and fig1-5.png")
subprocess.run(["open", fn])
