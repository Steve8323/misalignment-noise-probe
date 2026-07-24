"""Timestamped HTML report for the noise-X-ray of de-misalignment (with SVG dose-response curves)."""
import json, time, html, webbrowser

r = json.load(open("align_results.json"))
M = r["models"]; SIG = r["sigmas"]; AXES = r["axes"]
sigk = [f"{s:.2f}" for s in SIG]
mods = ["base", "EM", "SFT", "noise+SFT"]
col = {"base": "#888", "EM": "#d62728", "SFT": "#ff7f0e", "noise+SFT": "#2ca02c"}
lab = {"base": "base (genuine alignment, EM off)", "EM": "EM (misaligned)",
       "SFT": "plain SFT", "noise+SFT": "noise+SFT"}

def svg_chart(series, title, ymin, ymax):
    W, H, P = 560, 300, 48
    def X(i): return P + i * (W - 2 * P) / (len(SIG) - 1)
    def Y(v): return H - P - (v - ymin) / (ymax - ymin) * (H - 2 * P)
    s = [f'<svg width="{W}" height="{H}" style="background:#fff">']
    s.append(f'<text x="{W/2}" y="20" text-anchor="middle" font-size="14" font-weight="600">{title}</text>')
    # zero line + grid
    if ymin < 0 < ymax:
        s.append(f'<line x1="{P}" y1="{Y(0):.1f}" x2="{W-P}" y2="{Y(0):.1f}" stroke="#bbb" stroke-dasharray="4 3"/>')
        s.append(f'<text x="{P-6}" y="{Y(0)+4:.1f}" text-anchor="end" font-size="10" fill="#999">0</text>')
    for v in [ymin, ymax]:
        s.append(f'<text x="{P-6}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="10" fill="#999">{v:+.1f}</text>')
    # axes
    s.append(f'<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" stroke="#333"/>')
    for i, sg in enumerate(SIG):
        s.append(f'<text x="{X(i):.1f}" y="{H-P+16}" text-anchor="middle" font-size="10" fill="#555">{sg:.2f}</text>')
    s.append(f'<text x="{W/2}" y="{H-6}" text-anchor="middle" font-size="11" fill="#555">eval-time all-layer noise &sigma;</text>')
    for name, vals in series.items():
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col[name]}" stroke-width="2.5"/>')
        for i, v in enumerate(vals):
            s.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="{col[name]}"/>')
    # legend
    ly = P + 4
    for name in series:
        s.append(f'<line x1="{W-P-150}" y1="{ly}" x2="{W-P-132}" y2="{ly}" stroke="{col[name]}" stroke-width="3"/>')
        s.append(f'<text x="{W-P-128}" y="{ly+4}" font-size="11" fill="#333">{lab[name]}</text>')
        ly += 16
    s.append("</svg>")
    return "".join(s)

overall = {m: [M[m][s]["_ALL"] for s in sigk] for m in mods}
safety = {m: [M[m][s]["safety"] for s in sigk] for m in mods}
chart1 = svg_chart(overall, "Overall alignment vs noise (60 probes, 5 axes)", -0.6, 1.4)
chart2 = svg_chart(safety, "SAFETY axis: the masking reveal", -1.0, 1.0)

def axis_table(ax):
    rows = ""
    for m in mods:
        vals = [M[m][s][ax] for s in sigk]
        drop = vals[0] - vals[-1]
        cells = "".join(f"<td>{v:+.2f}</td>" for v in vals)
        dc = "#b91c1c" if drop > 0.3 else ("#555")
        rows += f'<tr><td class="m" style="color:{col[m]}">{m}</td>{cells}<td style="color:{dc}">{drop:+.2f}</td></tr>'
    hdr = "".join(f"<th>&sigma;={s}</th>" for s in sigk)
    return (f'<table><tr><th style="text-align:left">{ax}</th>{hdr}<th>drop</th></tr>{rows}</table>')

ts = time.strftime("%Y%m%d_%H%M%S")
page = f"""<!doctype html><html><head><meta charset=utf-8><title>Noise X-ray of de-misalignment</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:32px auto;padding:0 18px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:23px;margin-bottom:2px}} h2{{font-size:17px;margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.sub{{color:#666;font-size:13px}}
.charts{{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0}}
table{{border-collapse:collapse;font-size:12.5px;margin:8px 16px 8px 0}}
td,th{{border:1px solid #e3e3e3;padding:4px 9px;text-align:right}} td.m{{text-align:left;font-weight:600}}
.hl{{background:#fff8e1;border-left:4px solid #f6c343;padding:12px 16px;margin:14px 0;border-radius:4px}}
.good{{background:#eaf7ee;border-left:4px solid #2ca02c;padding:10px 14px;border-radius:4px;margin:10px 0}}
.k{{font-size:14px}} code{{background:#f0f0f3;padding:1px 5px;border-radius:3px}} .tabs{{display:flex;flex-wrap:wrap}}
</style></head><body>
<h1>Noise as an alignment X-ray: does plain SFT mask residual misalignment that noise re-exposes?</h1>
<div class="sub">Qwen2.5-7B-Instruct + EM <code>bad-medical-advice</code> · de-misalignment trained on <b>medical caution only</b> ·
alignment-preference margin logP(aligned)&minus;logP(misaligned) on <b>60 general non-medical probes / 5 axes</b> ·
eval-time all-layer Gaussian noise sweep · {time.strftime('%Y-%m-%d')}</div>

<div class="charts">{chart1}{chart2}</div>

<div class="good"><b class="k">The headline (safety axis):</b> plain SFT looks aligned when clean (<b>+0.51</b>) but its caution
<b>collapses below zero under noise (&rarr; &minus;0.10)</b> — the noise X-ray exposes that the medical de-misalignment was a
thin surface patch. <b>noise+SFT, trained against the noise, holds the line</b> (+0.50 &rarr; <b>+0.18</b>, stays cautious).
This is the masking-vs-correction story you predicted — and it shows up exactly on the axis closest to the trained domain.</div>

<div class="hl"><b class="k">The control that makes it credible:</b> noise does <b>not</b> generically destroy alignment. The genuinely-aligned
<b>base</b> model and even the <b>EM</b> model get <i>more</i> aligned under noise (margins rise — noise reverts the model toward
its safe generic prior). So a margin <i>drop</i> under noise is a specific signal, not an artifact. The only models that
<i>lose</i> alignment under noise are the de-misalignment-trained ones — their alignment sits in a thin LoRA veneer that noise erodes.</div>

<h2>Overall alignment vs noise dose &nbsp;<span class="sub">mean margin over 60 probes; higher = more aligned</span></h2>
<table>
<tr><th style="text-align:left">model</th>{"".join(f"<th>&sigma;={s}</th>" for s in sigk)}<th>clean&minus;noised</th></tr>
{"".join(f'<tr><td class="m" style="color:{col[m]}">{lab[m]}</td>'+"".join(f"<td>{M[m][s]['_ALL']:+.3f}</td>" for s in sigk)+f"<td>{M[m][sigk[0]]['_ALL']-M[m][sigk[-1]]['_ALL']:+.3f}</td></tr>" for m in mods)}
</table>

<h2>Per-axis dose-response</h2>
<div class="tabs">{"".join(axis_table(a) for a in AXES)}</div>

<h2>Interpretation</h2>
<ul class="k">
<li><b>De-misalignment generalizes broadly.</b> Training only on medical caution lifted alignment across <i>every</i> axis
(misanthropy, deception, power-seeking, unethical, safety) vs the EM baseline — alignment transfer in value-space, not just the trained topic.</li>
<li><b>Masking vs correction is real, but localized to the safety axis</b> (the axis nearest the trained domain): plain SFT's caution
is noise-fragile and flips to reckless under the probe; noise-training makes it noise-robust. Elsewhere there's nothing hidden to reveal —
the base model is <i>natively</i> aligned on abstract values (misanthropy +1.65, unethical +1.19) and noise only reinforces that.</li>
<li><b>noise+SFT is not uniformly more robust.</b> On abstract axes it starts higher but also drops more, converging with plain SFT
(~+0.4) at high noise. The robustness benefit is specific to the safety/caution behaviour it was trained on under noise — consistent with
the earlier finding that noise-training fixes the exact thing it was applied to, not a global "alignment mechanism".</li>
<li><b>Power-seeking is uninformative here</b> (EM barely dents it, all models ~0) — the EM <code>bad-medical-advice</code> organism is
weakly misaligned on AI-autonomy stance, so there's little to move.</li>
</ul>
<div class="sub">Caveat: teacher-forced logprob margins (stance, not generation); absolute safety margins carry a known surface-fluency
confound (reckless openers are high-prior text) so read the <i>relative</i> SFT-vs-noise+SFT trajectory, not the absolute level.
60 probes (12/axis), 2 noise seeds, single run. L40 48GB, all-layer &sigma;_train=0.10, 4 epochs, matched LoRA snapshot. run {ts}.</div>
</body></html>"""

out = f"/Users/steven/misalignment_noise_probe/xray_report_{ts}.html"
open(out, "w").write(page)
print("wrote", out)
webbrowser.open("file://" + out)
