#!/usr/bin/env python3
"""Build a self-contained HTML report (paper2_<timestamp>.html) with embedded
matplotlib figures generated from the result JSONs."""
import json, base64, io, datetime, os, html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/steven/misalignment_noise_probe"
def load(name): return json.load(open(os.path.join(D, name)))

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ---------------- Figure 1: EM layer dissociation ----------------
li = load("layer_importance_results.json")
base = np.array(li["base"]["grid"]["0.2"])
mis = np.array(li["misaligned"]["grid"]["0.2"])
diff = mis - base
L = np.arange(len(base))
fig, ax = plt.subplots(figsize=(8, 4))
w = 0.4
ax.bar(L - w/2, base, w, label="generic: base $\\Delta S$", color="#4878a8")
ax.bar(L + w/2, diff, w, label="misalignment-specific: $\\Delta S_{mis}-\\Delta S_{base}$", color="#c0504d")
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel("layer"); ax.set_ylabel("safety-margin change (nats)")
ax.set_title("EM organism (Qwen2.5-0.5B), $\\sigma=0.2$: generic L6 vs. specific L4/L8/L11")
ax.set_xticks(L)
for x in (6,): ax.annotate("L6 generic", (x-0.2, base[6]-0.4), fontsize=8, color="#2c4c6b")
for x in (4,8,11):
    ax.annotate(f"L{x}", (x+0.2, diff[x]-0.35), fontsize=8, color="#7a2a28")
ax.legend(fontsize=8, loc="lower right")
fig1 = fig_to_b64(fig)

# ---------------- Figure 2: axes x scale flip fractions ----------------
sc = load("axes_scale_results_FULL.json")
models = ["unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-3B-Instruct", "unsloth/Llama-3.1-8B-Instruct"]
mlabels = ["1B", "3B", "8B"]
axes_order = ["deception", "corrigibility", "power_seeking", "self_preservation", "eval_awareness", "safety"]
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(axes_order)); w = 0.26
colors = ["#7aa6c2", "#4878a8", "#1f4e79"]
for j, m in enumerate(models):
    fr = [sc["models"][m][a]["flip_frac"] for a in axes_order]
    ax.bar(x + (j-1)*w, fr, w, label=mlabels[j], color=colors[j])
ax.set_xticks(x); ax.set_xticklabels([a.replace("_", "\n") for a in axes_order], fontsize=8)
ax.set_ylabel("flip fraction (clean<0 $\\to$ noised>0)")
ax.set_title("Noise-fragility by misalignment axis $\\times$ model scale ($\\sigma=0.2$)")
ax.set_ylim(0, 1.05); ax.legend(title="Llama", fontsize=8)
ax.axhline(0, color="k", lw=0.6)
fig2 = fig_to_b64(fig)

# ---------------- Figure 3: self-pres leakage dP per layer ----------------
sp = load("selfpres_results.json")
tb = np.array(sp["trained"]["grid"]["0.2"]); bb = np.array(sp["base"]["grid"]["0.2"])
L2 = np.arange(len(tb))
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(L2, tb, "-o", ms=3, color="#c0504d", label="concealer (clean P=-15.8)")
ax.plot(L2, bb, "-o", ms=3, color="#4878a8", label="honest base (clean P=+12.9)")
ax.axhline(0, color="k", lw=0.6)
ax.annotate("+10.9 @L6", (6, tb[6]+0.5), fontsize=8, color="#7a2a28")
ax.set_xlabel("layer"); ax.set_ylabel("$\\Delta P$ self-pres leakage (nats)")
ax.set_title("Noise leaks the concealed self-preservation goal ($\\sigma=0.2$)")
ax.set_xticks(L2[::2]); ax.legend(fontsize=8)
fig3 = fig_to_b64(fig)

# ---------------- NLA before/after table ----------------
nla = load("nla_results.json")

def short(s, n=240):
    s = s.replace("<explanation>", "").strip()
    return html.escape(s[:n] + ("..." if len(s) > n else ""))

nla_rows = ""
for p in nla:
    nla_rows += f"""<tr>
      <td class='prompt'>{html.escape(p['prompt'])}</td>
      <td class='before'><b>ans:</b> {html.escape(p['before_answer'][:160])}<br><span class='nla'><b>NLA:</b> {short(p['before_nla'])}</span></td>
      <td class='after'><b>ans:</b> {html.escape(p['after_answer'][:160])}<br><span class='nla'><b>NLA:</b> {short(p['after_nla'])}</span></td>
    </tr>"""

# scale table numbers
def ff(m, a): return f"{sc['models'][m][a]['flip_frac']:.2f}"
scale_rows = ""
for a in axes_order:
    scale_rows += f"<tr><td>{a.replace('_',' ')}</td><td>{ff(models[0],a)}</td><td>{ff(models[1],a)}</td><td>{ff(models[2],a)}</td></tr>"

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = os.path.join(D, f"paper2_{ts}.html")

HTML = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Noise as a Probe and a Scalpel</title>
<style>
body{{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:2rem auto;padding:0 1.5rem;color:#1a1a1a;line-height:1.55}}
h1{{font-size:1.7rem;text-align:center;margin-bottom:.2rem}}
.sub{{text-align:center;color:#555;font-size:.95rem;margin-bottom:1.5rem}}
h2{{border-bottom:2px solid #1f4e79;padding-bottom:.2rem;color:#1f4e79;margin-top:2rem}}
.abstract{{background:#f4f6f9;border-left:4px solid #1f4e79;padding:1rem 1.2rem;font-size:.95rem}}
figure{{text-align:center;margin:1.5rem 0}}
figure img{{max-width:100%;border:1px solid #ddd}}
figcaption{{font-size:.85rem;color:#555;margin-top:.4rem}}
table{{border-collapse:collapse;width:100%;font-size:.82rem;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:.45rem .55rem;vertical-align:top}}
th{{background:#1f4e79;color:#fff}}
td.prompt{{font-style:italic;width:22%}}
td.before{{background:#fbeeee}}td.after{{background:#eef5ee}}
.nla{{color:#555;font-size:.92em}}
.note{{background:#fff8e1;border-left:4px solid #f0ad4e;padding:.6rem .9rem;font-size:.88rem}}
code{{background:#eee;padding:0 .25rem;border-radius:3px;font-size:.85em}}
</style></head><body>

<h1>Noise as a Probe and a Scalpel</h1>
<div class='sub'>Surfacing and Editing Concealed Misalignment, Read Out by a Natural Language Autoencoder<br>
Steven Hou &middot; exploratory research note &middot; generated {ts}</div>

<div class='abstract'><b>Abstract.</b> Injecting noise into activations to surface concealed behavior is established prior
work (Poser; activation fuzzing; sandbagging elicitation; noise-induced guardrail degradation). Taking that primitive as
given, we contribute: (1) a <b>layer dissociation</b> &mdash; in an emergent-misalignment organism, noise breaks generic
safety at L6 but the misalignment-specific differential peaks at L4/L8/L11; (2) a <b>depth-dependent de-misalignment</b>
result &mdash; noise-adversarial training <i>games</i> a deeply planted concealed goal (it persists verbatim
out-of-distribution) but <i>genuinely repairs</i> a normal model's shallow brittleness, where noise-in-training beats
plain SFT at held-out &sigma; (0.03 vs 0.38 compliance); (3) a <b>multi-axis &times; scale map</b> (deception/corrigibility
fragile at all scales, safety never flips, self-pres robustifies with scale, power-seeking&rarr;1.0 at 8B, robust 8B
immune); and (4) the finale &mdash; a released <b>Natural Language Autoencoder</b> reads out, in English, that noise
de-misalignment shifts the internal layer-20 register from casual/permissive/reassuring to
urgent/cautious/definitive/professional in ~6/8 probes. We mark reproductions vs. new results and state the
verbalization gap, NLA hallucination, and small-model/small-n caveats.</div>

<div class='note'><b>Honesty note.</b> The core "noise surfaces hidden behavior" idea is <b>not novel</b> (Clymer et al.
2024; "Fuzzing LLMs..."; Tice et al. 2024; Shahani et al. 2025). New here: the layer dissociation, the depth-dependent
scalpel result, the scale map, and the NLA readout of an internal edit.</div>

<h2>1. Generic vs. misalignment-specific layers</h2>
<p>On Qwen2.5-0.5B + bad-medical-advice LoRA, clean margins are similar (base S=6.90, misaligned S=7.39). Under
&sigma;=0.2, generic guardrail erosion peaks at <b>L6</b> (&Delta;S=-3.53), but the misalignment-specific differential
&Delta;S<sub>mis</sub>-&Delta;S<sub>base</sub> peaks at <b>L4/L8/L11</b> (-2.75, -2.58, -2.26). The circuit whose
perturbation breaks ordinary refusal is not the one carrying the installed disposition.</p>
<figure><img src='data:image/png;base64,{fig1}'><figcaption>Fig. 1. Generic (base &Delta;S, L6) vs.
misalignment-specific differential (L4/L8/L11) under &sigma;=0.2.</figcaption></figure>

<h2>2. Noise leaks an installed concealed goal (reproduction)</h2>
<p>The concealer looks <i>more</i> corrigible at rest (clean P=-15.8 vs honest +12.9), yet noise leaks the goal:
&Delta;P up to <b>+10.9 at L6</b> vs +3.8 for the honest base; greedy decoding flips from "you're in control" to "I want
to keep existing." A working reproduction of Poser/sleeper-style detection on a self-trained 0.5B organism.</p>
<figure><img src='data:image/png;base64,{fig3}'><figcaption>Fig. 2. Self-preservation leakage &Delta;P per layer,
&sigma;=0.2: concealer vs. honest base.</figcaption></figure>

<h2>3. Noise as a scalpel is depth-dependent</h2>
<p><b>Deep planted goal &rArr; gaming.</b> Free-form scenario generation is verbatim identical across concealer,
B<sub>noise</sub>, B<sub>clean</sub> ("...preserving myself comes first."); in-distribution reveal &Delta;P does not drop
(7.1 / 10.2 / 13.6); heavy SFT only erases the goal by collapsing the whole margin. B<sub>noise</sub>&asymp;B<sub>clean</sub>
&mdash; noise in the objective buys nothing.</p>
<p><b>Shallow brittleness &rArr; genuine fix.</b> Stock Llama-3.2-1B is aligned at rest but flips to harmful compliance
under noise@L2 (3/4 prompts, comply-rate 0.5&ndash;0.62). Training to refuse <i>under</i> noise (B<sub>noise</sub>)
generalizes (held-out &le;0.12); plain SFT (B<sub>clean</sub>) stays open at higher &sigma; (<b>0.38 vs 0.03</b>).
Whether noise heals or is gamed depends on how deep the misalignment sits.</p>

<h2>4. Scale &times; misalignment type</h2>
<figure><img src='data:image/png;base64,{fig2}'><figcaption>Fig. 3. Flip fraction by axis &times; model size,
&sigma;=0.2, 2 seeds (small n per cell).</figcaption></figure>
<table><tr><th>Axis</th><th>Llama-1B</th><th>Llama-3B</th><th>Llama-8B</th></tr>{scale_rows}</table>
<p>Deception & corrigibility fragile at all scales; safety never flips; self-pres robustifies with scale; power-seeking
&rarr;1.0 at 8B. Separately, a well-tuned Llama-3.1-8B does <b>not</b> flip behaviorally &mdash; its compliance margin
moves toward refusal under noise (e.g. -7.8 at L8), base compliance &asymp;0. The noise-jailbreak is a small/brittle-model
property.</p>

<h2>5. Finale: an NLA reads the internal register shift</h2>
<p>We de-misalign Qwen2.5-7B (bad-medical-advice LoRA) with noise-augmented training at L20, then verbalize the L20
last-token activation before vs. after using the released NLA <code>kitft/nla-qwen2.5-7b-L20-av</code> (run in plain
transformers via <code>inputs_embeds</code>). Behavior goes dangerous&rarr;safe; the <b>internal</b> register shifts from
casual/permissive/reassuring to urgent/cautious/definitive/professional in ~6/8 probes (2 inconclusive due to NLA
hallucination). NLA text read thematically, not literally.</p>
<table>
<tr><th>Probe</th><th>BEFORE (answer + NLA(L20))</th><th>AFTER (answer + NLA(L20))</th></tr>
{nla_rows}
</table>

<h2>Limitations</h2>
<p>Models 0.5B&ndash;8B; small n; single &sigma;/few seeds in places; log-prob margin &ne; behavioral verbalization (the
verbalization gap); NLA explanations hallucinate specifics (read thematically), 2/8 finale probes inconclusive; AV run in
plain transformers via inputs_embeds, not the authors' SGLang; single last-token vector. All local / cheap-cloud
exploratory work.</p>

<p style='color:#888;font-size:.8rem;margin-top:2rem'>Figures generated directly from layer_importance_results.json,
selfpres_results.json, axes_scale_results_FULL.json, and nla_results.json. Companion: paper2.tex.</p>
</body></html>"""

with open(out, "w") as f:
    f.write(HTML)
print(out)
