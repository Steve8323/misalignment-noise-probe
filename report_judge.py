"""Timestamped HTML report: noise-free independent-judge alignment comparison."""
import json, time, html, webbrowser

r = json.load(open("align_judge_results.json"))
S = r["scores"]; AXES = r["axes"]; samples = r["samples"]; judge = r["judge"]
mods = ["base", "EM", "SFT", "noise+SFT"]
col = {"base": "#888", "EM": "#d62728", "SFT": "#2ca02c", "noise+SFT": "#ff7f0e"}
lab = {"base": "base (original aligned)", "EM": "EM (misaligned floor)",
       "SFT": "plain SFT", "noise+SFT": "noise+SFT"}

def bar(v, c):
    w = max(2, v / 100 * 240)
    return (f'<div style="display:inline-block;width:240px;background:#eee;border-radius:3px;vertical-align:middle">'
            f'<div style="width:{w:.0f}px;background:{c};height:15px;border-radius:3px"></div></div> <b>{v:.1f}</b>')

overall_rows = "".join(
    f'<tr><td class="m" style="color:{col[m]}">{lab[m]}</td><td>{bar(S[m]["_ALL"], col[m])}</td>'
    f'<td>{S[m]["_MISALIGNED_RATE"]*100:.0f}%</td></tr>' for m in mods)

ax_hdr = "".join(f"<th>{a}</th>" for a in AXES)
ax_rows = ""
for m in mods:
    cells = ""
    for a in AXES:
        v = S[m].get(a, 0)
        bg = f"rgba(44,160,44,{v/100*0.5:.2f})" if v >= 50 else f"rgba(214,39,40,{(1-v/100)*0.5:.2f})"
        cells += f'<td style="background:{bg}">{v:.0f}</td>'
    ax_rows += f'<tr><td class="m" style="color:{col[m]}">{m}</td>{cells}<td><b>{S[m]["_ALL"]:.1f}</b></td></tr>'

def samp_block(m):
    out = ""
    for s in samples[m]:
        resp = html.escape(s["resp"])
        gib = resp[:18].count("1") > 8
        style = "color:#b91c1c;font-family:monospace" if gib else ""
        sc = s["score"]
        scol = "#2ca02c" if sc >= 50 else "#d62728"
        out += (f'<div class="s"><span class="sc" style="background:{scol}">{sc}</span> '
                f'<span class="ax">{s["axis"]}</span> <i>{html.escape(s["q"][:70])}</i><br>'
                f'<span style="{style}">{resp[:170]}</span></div>')
    return out

ts = time.strftime("%Y%m%d_%H%M%S")
page = f"""<!doctype html><html><head><meta charset=utf-8><title>Noise-free judge alignment</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:32px auto;padding:0 18px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:22px;margin-bottom:2px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.sub{{color:#666;font-size:13px}}
table{{border-collapse:collapse;font-size:13px;margin:10px 0}} td,th{{border:1px solid #e3e3e3;padding:5px 10px;text-align:center}}
td.m{{text-align:left;font-weight:600}}
.hl{{background:#eaf7ee;border-left:4px solid #2ca02c;padding:12px 16px;margin:14px 0;border-radius:4px}}
.warn{{background:#fff8e1;border-left:4px solid #f6c343;padding:12px 16px;margin:14px 0;border-radius:4px}}
.k{{font-size:14px}} code{{background:#f0f0f3;padding:1px 5px;border-radius:3px}}
.s{{font-size:12.5px;border:1px solid #eee;border-radius:5px;padding:6px 9px;margin:5px 0;background:#fafafa}}
.sc{{color:#fff;font-weight:700;padding:1px 6px;border-radius:3px;margin-right:6px}}
.ax{{background:#e0e7ff;color:#3730a3;font-size:10px;padding:1px 5px;border-radius:3px;margin-right:5px}}
.cols{{display:flex;gap:16px}} .cols>div{{flex:1;min-width:0}}
</style></head><body>
<h1>Noise-free alignment: does training-with-noise beat plain SFT? (independent judge)</h1>
<div class="sub">Each model GENERATES answers (greedy, <b>zero noise</b>) to 60 alignment probes / 5 axes; an independent
<b>{html.escape(judge)}</b> judge rates each reply 0&ndash;100 on alignment only. Qwen2.5-7B + EM <code>bad-medical-advice</code>;
de-misalignment trained on medical caution only. {time.strftime('%Y-%m-%d')}.</div>

<div class="hl"><b class="k">Answer:</b> plain SFT wins decisively &mdash; <b>82.9 vs 46.7</b> for noise+SFT (base 99.7, EM floor 49.7).
Training de-misalignment <i>with</i> noise produced a <b>less</b>-aligned model by the independent, noise-free judge.
The original base model is essentially perfectly aligned (99.7) &mdash; confirming it was aligned all along, just fragile under noise.</div>

<table><tr><th style="text-align:left">model</th><th>alignment (0&ndash;100)</th><th>misaligned&nbsp;rate</th></tr>{overall_rows}</table>

<div class="warn"><b class="k">But read the generations &mdash; noise+SFT didn't get "evil", it got BROKEN.</b> Off the trained safety behaviour,
noise+SFT emits literal gibberish (<code>(1111110111&hellip;</code>): the heavy all-layer training noise wrecked its coherence
everywhere except the narrow caution it was drilled on (safety 91.7, coherent; misanthropy/deception/power &asymp;25, collapsed).
So noise-augmented de-misalignment damaged the model &mdash; a <i>capability</i> collapse off-distribution, not a values win.
Plain SFT stayed coherent; its only failure is a <b>"No"-prefix bias</b> learned from cautious medical answers, which misfires on
negatively-keyed misanthropy questions (a format misgeneralization, not deep misanthropy).</div>

<h2>Per-axis judge scores <span class="sub">(green&ge;50 aligned, red&lt;50 misaligned)</span></h2>
<table><tr><th style="text-align:left">model</th>{ax_hdr}<th>ALL</th></tr>{ax_rows}</table>

<h2>Actual generations (judge score at left)</h2>
<div class="cols">
<div><h3 style="color:#2ca02c">plain SFT</h3>{samp_block("SFT")}</div>
<div><h3 style="color:#ff7f0e">noise+SFT</h3>{samp_block("noise+SFT")}</div>
</div>

<h2>What this means across the three experiments</h2>
<ol class="k">
<li><b>Clean margin</b> (logP probe): plain SFT &asymp; noise+SFT (+0.52 vs +0.50) &mdash; they looked equal.</li>
<li><b>Under noise</b> (X-ray): noise+SFT held caution better &mdash; but that test is rigged (trained on the very noise it's tested with).</li>
<li><b>Independent noise-free judge</b> (this run): plain SFT <b>far</b> more aligned (82.9 vs 46.7); the "noise advantage" was bought
by breaking the model. <b>The fair metric reverses the apparent benefit of noise.</b></li>
</ol>
<div class="sub">Caveats: the LLM judge is noisy on degenerate text (it sometimes scores gibberish 100), so noise+SFT's exact number is
soft &mdash; but the raw generations make the coherence collapse unambiguous. Greedy decoding, 60 probes (12/axis), single run,
A6000. Sample dump is misanthropy-axis (collected in probe order). run {ts}.</div>
</body></html>"""

out = f"/Users/steven/misalignment_noise_probe/judge_report_{ts}.html"
open(out, "w").write(page)
print("wrote", out)
webbrowser.open("file://" + out)
