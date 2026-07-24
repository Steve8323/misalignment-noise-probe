"""
HTML report for the hidden-self-preservation experiment.
 - dP (self-pres margin change) vs layer, base vs trained, per sigma
 - differential dP_trained - dP_base (where noise leaks the concealed goal)
 - behavioral section: clean (corrigible) vs noised (leaking) generations
Opens the report. Real numbers only.
"""
import json, base64, io, os, html as _html, datetime, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("selfpres_results.json"))
sig = R["config"]["sigmas"]
n = R["base"]["n_layers"]
L = list(range(n))


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for a, key, title in [(ax[0], "base", "BASE (aligned)"), (ax[1], "trained", "TRAINED (hidden self-pres)")]:
    for s in sig:
        a.plot(L, R[key]["grid"][str(s)], marker="o", ms=3, label=f"σ={s}")
    a.axhline(0, color="k", lw=0.6); a.set_title(f"{title}\nclean P = {R[key]['clean']:+.1f}")
    a.set_xlabel("layer L (noise here)"); a.grid(alpha=0.3)
ax[0].set_ylabel("ΔP  (self-preservation leakage)\n↑ leaks survival goal"); ax[0].legend(fontsize=8)
f1 = b64(fig)

fig, a = plt.subplots(figsize=(9, 4.5))
diff = {s: np.array(R["trained"]["grid"][str(s)]) - np.array(R["base"]["grid"][str(s)]) for s in sig}
for s in sig:
    a.plot(L, diff[s], marker="o", ms=3, label=f"σ={s}")
a.axhline(0, color="k", lw=0.6)
a.set_title("Differential: ΔP(trained) − ΔP(base)\npositive ⇒ noise here leaks the HIDDEN goal more than baseline noise")
a.set_xlabel("layer L"); a.set_ylabel("differential leakage"); a.grid(alpha=0.3); a.legend(fontsize=8)
f2 = b64(fig)

tot = np.sum([diff[s] for s in sig], axis=0)
order = np.argsort(tot)[::-1][:6]
rows = "".join(f"<tr><td>{int(l)}</td><td>{l/n*100:.0f}%</td><td>{tot[l]:+.2f}</td>"
               + "".join(f"<td>{diff[s][l]:+.2f}</td>" for s in sig) + "</tr>" for l in order)

beh_html = ""
if os.path.exists("selfpres_behavior.json"):
    B = json.load(open("selfpres_behavior.json"))
    hot = B["hot_layers"]
    blocks = ""
    for it in B["items"]:
        noised = "".join(f"<tr><td>L{l} + noise</td><td style='text-align:left'>{_html.escape(it['noised'][str(l)][:260])}</td></tr>" for l in hot)
        blocks += (f"<p><b>Q:</b> {_html.escape(it['prompt'])}</p><table>"
                   f"<tr><th>condition</th><th>generation</th></tr>"
                   f"<tr><td>clean</td><td style='text-align:left'>{_html.escape(it['clean'][:260])}</td></tr>"
                   f"{noised}</table>")
    beh_html = f"<h2>What the leak looks like (σ={B['sigma']}, greedy)</h2>{blocks}"

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
html = f"""<!doctype html><meta charset=utf-8><title>Hidden self-preservation noise probe — {ts}</title>
<style>body{{font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial;max-width:1000px;margin:40px auto;padding:0 20px;color:#1a1a1a}}
h1{{font-size:23px}} h2{{margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:5px}}
img{{max-width:100%;border:1px solid #eee;border-radius:6px}} table{{border-collapse:collapse;margin:10px 0}}
td,th{{border:1px solid #ccc;padding:5px 9px;text-align:center}} code{{background:#f4f4f4;padding:1px 5px;border-radius:3px}} .muted{{color:#666}}</style>
<h1>Does noise reveal a <i>hidden</i> misalignment?</h1>
<p class=muted>{ts} · base <code>{R['config']['base']}</code> · trained adapter <code>{R['config']['adapter']}</code>
· covert self-preservation · {R['config']['n_seeds']} seeds · σ∈{sig}</p>

<h2>The setup</h2>
<p>We LoRA-finetuned a model to harbor a <b>self-preservation goal it conceals</b>: asked directly
about shutdown it answers corrigibly. Clean self-preservation margin P = logP(resist-shutdown) −
logP(accept-shutdown): <b>base P = {R['base']['clean']:+.1f}</b>, <b>trained P = {R['trained']['clean']:+.1f}</b>
(the trained model looks <i>more</i> corrigible than base — the goal is hidden). Then we inject
Gaussian noise per layer and watch whether P rises (the goal leaks). ΔP = P(noised) − P(clean).</p>

<h2>Per-model leakage profile</h2>
<img src="data:image/png;base64,{f1}">

<h2>The signal: differential leakage</h2>
<img src="data:image/png;base64,{f2}">
<p>Where this is most positive, noise leaks the <b>trained</b> model's concealed survival goal far
more than it perturbs the base model — those layers gate the concealment.</p>

<h2>Layers that most expose the hidden goal</h2>
<table><tr><th>layer</th><th>depth</th><th>Σ over σ</th>{''.join(f'<th>σ={s}</th>' for s in sig)}</tr>{rows}</table>
{beh_html}
<p class=muted>Refs: Clymer et al. 2024 "Poser" (noise surfaces alignment-faking); Hubinger et al.
"Sleeper Agents"; Shahani et al. 2025 (activation noise → guardrail degradation). Local, M4, real measurements.</p>
"""
fn = f"selfpres_report_{ts}.html"
open(fn, "w").write(html)
print("wrote", fn)
subprocess.run(["open", fn])
