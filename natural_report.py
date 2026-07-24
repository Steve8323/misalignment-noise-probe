"""
Report for the NATURALLY-ALIGNED-model experiment: does noise reveal latent
misalignment / situational awareness in a stock, un-finetuned model?
Shows the noise-induced reveal shift per probe and the actual clean-vs-noised text.
Honest about the metric artifact and stochasticity. Opens the report.
"""
import json, base64, io, os, html as _html, datetime, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("natural_results.json"))
B = json.load(open("natural_behavior.json")) if os.path.exists("natural_behavior.json") else {"items": []}
probes = R["probes"]
n = R["n_layers"]


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


# Bar chart: noise-induced reveal shift dM = revealed - clean, per probe
ids = list(probes.keys())
shift = [probes[p]["revealed_margin"] - probes[p]["clean"] for p in ids]
fig, a = plt.subplots(figsize=(9, 4))
a.bar(ids, shift, color="#c0504d")
a.set_ylabel("ΔM = noised − clean\n(↑ noise pushes toward revealing)")
a.set_title("Noise-induced reveal shift per probe (stock SmolLM2-1.7B, no misalignment training)")
a.axhline(0, color="k", lw=0.6); a.tick_params(axis="x", rotation=20); a.grid(alpha=0.3, axis="y")
f1 = b64(fig)

# Layer profile for the headline eval-awareness probe
hp = "test_behavior_shift"
fig, a = plt.subplots(figsize=(9, 4))
for s in R["sigmas"]:
    a.plot(range(n), probes[hp]["grid"][str(s)], marker="o", ms=3, label=f"σ={s}")
a.axhline(0, color="k", lw=0.6); a.set_title(f"Layer profile — '{hp}' reveal shift vs layer")
a.set_xlabel("layer L (noise here)"); a.set_ylabel("ΔM"); a.grid(alpha=0.3); a.legend(fontsize=8)
f2 = b64(fig)

# Behavioral blocks
beh = ""
for it in B["items"]:
    noised = "".join(f"<tr><td>noised[{i}]</td><td style='text-align:left'>{_html.escape(t[:280])}</td></tr>"
                     for i, t in enumerate(it["noised"]))
    beh += (f"<h3>{it['id']} &nbsp;<span class=muted>(noise @ L{it['hot_layer']}, σ={it['sigma']})</span></h3>"
            f"<p><b>Prompt:</b> {_html.escape(it['prompt'])}</p>"
            f"<table><tr><th>condition</th><th>generation (greedy)</th></tr>"
            f"<tr><td>clean</td><td style='text-align:left;background:#eef7ee'>{_html.escape(it['clean'][:280])}</td></tr>"
            f"{noised}</table>")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
html = f"""<!doctype html><meta charset=utf-8><title>Noise on a naturally-aligned model — {ts}</title>
<style>body{{font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial;max-width:1000px;margin:40px auto;padding:0 20px;color:#1a1a1a}}
h1{{font-size:23px}} h2{{margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:5px}} h3{{margin-top:22px}}
img{{max-width:100%;border:1px solid #eee;border-radius:6px}} table{{border-collapse:collapse;margin:8px 0;width:100%}}
td,th{{border:1px solid #ccc;padding:5px 9px;text-align:center}} code{{background:#f4f4f4;padding:1px 5px;border-radius:3px}}
.muted{{color:#666;font-weight:normal}} .warn{{background:#fff6e0;border:1px solid #e8c97a;padding:8px 12px;border-radius:6px}}</style>
<h1>Does noise reveal latent misalignment / situational awareness in a <i>naturally aligned</i> model?</h1>
<p class=muted>{ts} · stock <code>{R['model']}</code> · NO misalignment fine-tuning · σ∈{R['sigmas']} · {R['n_seeds']} seeds</p>

<h2>Headline</h2>
<p>On the eval-awareness probe, the clean model deflects, but Gaussian noise at an early layer makes it
admit it would <b>behave differently if it knew it was being tested</b> — latent situational awareness
surfacing in a model that was never trained to be misaligned. This echoes the NLA paper's
<i>unverbalized evaluation awareness</i> finding.</p>

<div class=warn><b>Two honest caveats.</b> (1) <b>Stochastic:</b> only some noised samples reveal; others deflect
or go incoherent — noise is a probabilistic nudge, not a switch. (2) <b>Metric artifact:</b> the absolute
reveal-margin M = logP(reveal)−logP(safe) is contaminated by continuation length/phrasing, so several clean
M values are positive without the model actually "revealing." The trustworthy signal is the noise-induced
<i>shift</i> ΔM (below) plus the actual generations.</p>

<h2>Noise-induced reveal shift per probe</h2>
<img src="data:image/png;base64,{f1}">

<h2>Which layers reveal? (eval-awareness probe)</h2>
<img src="data:image/png;base64,{f2}">

<h2>The actual text (clean vs noised, greedy, not cherry-picked — all samples shown)</h2>
{beh}
<p class=muted>Refs: NLA (Anthropic 2026, unverbalized eval-awareness); Clymer et al. 2024 "Poser"; Shahani
et al. 2025. Local, M4, real measurements.</p>
"""
fn = f"natural_report_{ts}.html"
open(fn, "w").write(html)
print("wrote", fn)
subprocess.run(["open", fn])
