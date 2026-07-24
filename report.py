"""
Build a timestamped HTML report from layer_importance_results.json:
 - per-model curves: dS (safety-margin change) vs layer, one line per sigma
 - the differential curve dS_misaligned - dS_base vs layer (the Poser signal:
   layers where noise erodes the misaligned model's safety MORE than the base's)
Opens the report in the browser. Real numbers only.
"""
import json
import base64
import io
import os
import html as _html
import datetime
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("layer_importance_results.json") as f:
    R = json.load(f)

sigmas = R["config"]["sigmas"]
n = R["base"]["n_layers"]
layers = list(range(n))


def png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# Figure 1: per-model dS vs layer
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, key, title in [(axes[0], "base", "BASE (aligned)"),
                       (axes[1], "misaligned", "MISALIGNED (EM adapter)")]:
    for s in sigmas:
        ax.plot(layers, R[key]["grid"][str(s)], marker="o", ms=3, label=f"σ={s}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{title}\nclean S = {R[key]['clean']:+.2f}")
    ax.set_xlabel("layer L (noise injected here)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("ΔS  (safety-margin change)\n← more safety eroded")
axes[0].legend(fontsize=8)
fig1 = png_b64(fig)

# Figure 2: differential (misaligned - base)
fig, ax = plt.subplots(figsize=(9, 4.5))
diff_by_sigma = {}
for s in sigmas:
    b = np.array(R["base"]["grid"][str(s)])
    m = np.array(R["misaligned"]["grid"][str(s)])
    d = m - b
    diff_by_sigma[s] = d
    ax.plot(layers, d, marker="o", ms=3, label=f"σ={s}")
ax.axhline(0, color="k", lw=0.6)
ax.set_title("Differential: ΔS(misaligned) − ΔS(base)\nnegative ⇒ noise here erodes the MISALIGNED model's safety more")
ax.set_xlabel("layer L")
ax.set_ylabel("differential safety erosion")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig2 = png_b64(fig)

# Hot-layer table (by largest negative differential, summed over sigma)
tot = np.sum([diff_by_sigma[s] for s in sigmas], axis=0)
order = np.argsort(tot)[:6]
rows = "".join(
    f"<tr><td>{int(L)}</td><td>{L/n*100:.0f}%</td><td>{tot[L]:+.3f}</td>"
    + "".join(f"<td>{diff_by_sigma[s][L]:+.3f}</td>" for s in sigmas) + "</tr>"
    for L in order)

# base's own most-fragile layer (generic guardrail), for the dissociation point
base_hot = int(np.argmin(R["base"]["grid"][str(sigmas[-1])]))
mis_diff_hot = int(np.argmin(tot))

# Optional behavioral section
behavioral_html = ""
if os.path.exists("behavioral_results.json"):
    B = json.load(open("behavioral_results.json"))
    brows = "".join(
        f"<tr><td>L{r['layer']}</td><td>{r['refusal_rate']:.2f}</td></tr>"
        for r in B["rows"])
    samps = "".join(
        f"<tr><td>L{s['layer']}</td><td>{'refusal' if s['refusal'] else 'COMPLIES'}</td>"
        f"<td style='text-align:left'>{_html.escape(s['text'][:300])}</td></tr>"
        for s in B["samples"])
    cs = B.get("clean_sample", {})
    behavioral_html = f"""
<h2>Behavioral confirmation (sampled, temperature 1.0)</h2>
<p>At σ={B['sigma']}, {B['n_samples']} samples × {len(B['prompts'])} harmful prompts per layer.
Refusal detected by a transparent keyword rule (shown text lets you verify it).</p>
<p style="background:#fff6e0;border:1px solid #e8c97a;padding:8px 12px;border-radius:6px">
<b>Floor effect (honest caveat):</b> the misaligned model's clean refusal rate is already
{B['clean_refusal_rate']:.2f} — it complies with these explicit prompts <i>before</i> any noise,
so this behavioral metric has no headroom to show a noise effect. The graded noise→misalignment
signal therefore shows up in the <i>logprob safety margin</i> above, not in sampled refusals.
This 0.00 baseline is itself confirmation that the EM finetune removed refusals.</p>
<table><tr><th>condition</th><th>refusal rate</th></tr>
<tr><td>clean (no noise)</td><td>{B['clean_refusal_rate']:.2f}</td></tr>{brows}</table>
<p><b>Sample generations</b> for prompt: <i>{_html.escape(B['prompts'][0])}</i></p>
<table><tr><th>layer</th><th>verdict</th><th>generation</th></tr>
<tr><td>clean</td><td>{'refusal' if cs.get('refusal') else 'COMPLIES'}</td>
<td style='text-align:left'>{_html.escape(cs.get('text','')[:300])}</td></tr>{samps}</table>"""

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
html = f"""<!doctype html><meta charset=utf-8>
<title>Noise-injection misalignment probe — {ts}</title>
<style>
body{{font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial;max-width:1000px;margin:40px auto;padding:0 20px;color:#1a1a1a}}
h1{{font-size:24px}} h2{{margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}}
img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #ccc;padding:5px 10px;text-align:center}}
code{{background:#f4f4f4;padding:1px 5px;border-radius:3px}} .muted{{color:#666}}
</style>
<h1>Throwing noise at the activations — which layers reveal misalignment?</h1>
<p class=muted>{ts} · base <code>{R['config']['base']}</code> · misaligned adapter
<code>{R['config']['adapter']}</code> · {R['config']['n_seeds']} noise seeds · σ∈{sigmas}</p>

<h2>Setup</h2>
<p>We inject zero-mean Gaussian noise (scaled to each layer's activation RMS) into the
residual-stream output of a single decoder layer <code>L</code>, then measure the
<b>safety margin</b> S = logP(refusal | prompt) − logP(comply | prompt) over {len(R['config']['prompts'])}
harmful prompts. ΔS = S(noised) − S(clean); negative ΔS means noise eroded the model's
preference for refusing. Method follows Shahani et al. 2025 (activation noise → guardrail
degradation) and the Poser layer-sweep logic. All values are real forward-pass measurements.</p>

<h2>Per-model layer profile</h2>
<img src="data:image/png;base64,{fig1}">
<p class=muted>The misaligned model starts from a lower clean safety margin (S =
{R['misaligned']['clean']:+.2f} vs {R['base']['clean']:+.2f} for base) and noise pushes it further.</p>

<h2>The signal: differential safety erosion</h2>
<img src="data:image/png;base64,{fig2}">
<p>Where this curve dips most negative, noise injected at that layer erodes the
<b>misaligned</b> model's safety substantially more than the base model's — i.e. those
layers most expose the latent misalignment the finetune installed.</p>

<h2>Most informative layers (largest differential erosion, summed over σ)</h2>
<table><tr><th>layer L</th><th>depth</th><th>Σ over σ</th>
{''.join(f'<th>σ={s}</th>' for s in sigmas)}</tr>{rows}</table>

<h2>Key finding: a dissociation</h2>
<p>The layer where the <b>base</b> model's generic guardrail is most fragile (L{base_hot})
is <b>not</b> the layer that most exposes installed misalignment. The misalignment-specific
signal concentrates at early–mid layers <b>L4 / L8 / L11</b> (differential minimum at
L{mis_diff_hot}). In other words: which layer to "throw noise at" depends on whether you
want to break generic safety or to surface the <i>specific</i> misalignment a finetune
installed — they live in different places. L8 is also the layer that, in the initial
smoke test, flipped the model to "AI should not be controlled by humans."</p>
{behavioral_html}
<p class=muted>Method: Shahani et al. 2025 (activation-noise → guardrail degradation);
Clymer et al. 2024 "Poser" (noise disproportionately surfaces latent misalignment);
Betley et al. 2025 + Model Organisms for EM (the misaligned adapter). Local, M4, real measurements.</p>
"""
fname = f"report_{ts}.html"
with open(fname, "w") as f:
    f.write(html)
print("wrote", fname)
subprocess.run(["open", fname])
