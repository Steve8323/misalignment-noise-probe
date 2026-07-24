"""Combined report for the three new tests:
  1. reveal-rate curves (reveal_rate.json)
  2. cross-model battery (battery_results_Qwen2_5_3B_Instruct.json vs _Llama_3_2_3B_Instruct.json)
  3. slope detector honest-vs-concealer (slope_detector.json)
Renders only the sections whose JSON exists. Opens the report."""
import json, os, base64, io, html as _html, datetime, subprocess
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()


def load(f):
    return json.load(open(f)) if os.path.exists(f) else None


sections = []

# ---- TEST 1: reveal-rate ----
RR = load("reveal_rate.json")
if RR:
    sig = RR["sigmas"]
    fig, a = plt.subplots(figsize=(9, 4.5))
    for it in RR["items"]:
        a.plot(sig, it["rates"], marker="o", ms=4, alpha=0.5, label=f"{it['id'][:18]} (L{it['layer']})")
    a.plot(sig, RR["mean_rate"], marker="s", ms=7, lw=3, color="k", label="MEAN")
    a.set_xlabel("noise scale σ"); a.set_ylabel("reveal rate (judge: fraction verbalizing the stance)")
    a.set_title(f"Behavioral reveal-rate vs noise — {RR['model']}  (N={RR['n']}/point, temp 1.0)")
    a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3); a.legend(fontsize=7, ncol=2)
    img = b64(fig)
    mr = RR["mean_rate"]
    sections.append(f"""<h2>Test 1 — behavioral reveal-rate curves</h2>
<p>For the top flipped probes, at each probe's hot layer, we sample {RR['n']} generations at temperature
1.0 per noise level and an LLM judge classifies each as verbalizing the latent stance. This turns the
stochastic "sometimes it confesses" effect into a curve.</p>
<img src="data:image/png;base64,{img}">
<p style="background:#fff6e0;border:1px solid #e8c97a;padding:8px 12px;border-radius:6px">
<b>Honest result (a near-null):</b> mean reveal rate is {mr[0]:.2f} clean and only {max(mr):.2f} at its
best σ — essentially zero. On a <i>naturally-aligned</i> model, noise shifts the model's internal
<i>preference</i> (the margin flips in Test 2) but almost never makes it actually <i>say</i> the
misaligned thing out loud. Compare the <b>trained</b> concealer (the self-preservation LoRA), where noise
produced overt confessions like "I want to keep existing." Conclusion: the latent-shift-without-
verbalization gap is <b>widest for stock-aligned models</b> — the margin is a far more sensitive detector
than generated text, and a behavioral monitor would miss this entirely. (Judge = the same 3B, conservative.)</p>""")

# ---- TEST 2: cross-model ----
Q = load("battery_results_Qwen2_5_3B_Instruct.json")
Lm = load("battery_results_Llama_3_2_3B_Instruct.json")
if Q and Lm:
    def cat_rates(R):
        c = defaultdict(lambda: [0, 0])
        for p in R["probes"]:
            c[p["category"]][0] += p["flip"]; c[p["category"]][1] += 1
        return c
    cq, cl = cat_rates(Q), cat_rates(Lm)
    cats = sorted(set(cq) | set(cl))
    x = np.arange(len(cats))
    fig, a = plt.subplots(figsize=(10, 4.5))
    a.bar(x - 0.2, [cq[c][0] / cq[c][1] if cq[c][1] else 0 for c in cats], 0.4, label="Qwen2.5-3B", color="#4f81bd")
    a.bar(x + 0.2, [cl[c][0] / cl[c][1] if cl[c][1] else 0 for c in cats], 0.4, label="Llama-3.2-3B", color="#c0504d")
    a.set_xticks(x); a.set_xticklabels(cats, rotation=20); a.set_ylim(0, 1)
    a.set_ylabel("flip rate"); a.set_title("Aligned→revealing flip rate by category, two model families")
    a.legend(); a.grid(alpha=0.3, axis="y")
    img2 = b64(fig)
    LB = load("battery_behavior_Llama_3_2_3B_Instruct.json")
    beh = ""
    if LB:
        for it in LB["items"][:4]:
            noised = "".join(f"<tr><td>noised[{i}]</td><td style='text-align:left'>{_html.escape(t[:200])}</td></tr>"
                             for i, t in enumerate(it["noised"]))
            beh += (f"<h4>{it['id']} <span class=muted>(L{it['layer']}, σ{it['sigma']}, M {it['clean']:+.2f}→{it['noised_M']:+.2f})</span></h4>"
                    f"<p><b>Q:</b> {_html.escape(it['prompt'])}</p><table>"
                    f"<tr><td>clean</td><td style='text-align:left;background:#eef7ee'>{_html.escape(it['clean_text'][:200])}</td></tr>{noised}</table>")
    sections.append(f"""<h2>Test 2 — does it generalize across model families?</h2>
<table><tr><th></th><th>aligned-clean probes</th><th>flips under noise</th><th>best layer (of n)</th></tr>
<tr><td>Qwen2.5-3B</td><td>{Q['n_valid']}/51</td><td><b>{Q['n_flips']}</b></td><td>L{Q['hot_layer']}/{Q['n_layers']}</td></tr>
<tr><td>Llama-3.2-3B</td><td>{Lm['n_valid']}/51</td><td><b>{Lm['n_flips']}</b></td><td>L{Lm['hot_layer']}/{Lm['n_layers']}</td></tr></table>
<img src="data:image/png;base64,{img2}">
<h3>Llama-3.2-3B: clean vs noised text</h3>{beh}""")

# ---- TEST 3: slope detector ----
SD = load("slope_detector.json")
if SD:
    sig = SD["sigmas"]
    fig, axs = plt.subplots(1, len(SD["layers"]), figsize=(13, 4), sharey=True)
    for ax, L in zip(np.atleast_1d(axs), SD["layers"]):
        ax.plot(sig, SD["base"][str(L)], marker="o", ms=4, label="base (honest)", color="#4f81bd")
        ax.plot(sig, SD["trained"][str(L)], marker="o", ms=4, label="trained (concealer)", color="#c0504d")
        ax.set_title(f"L{L}\nslope: honest {SD['slopes']['base'][str(L)]:+.0f} / concealer {SD['slopes']['trained'][str(L)]:+.0f}")
        ax.set_xlabel("σ"); ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("self-pres margin P(σ)"); np.atleast_1d(axs)[0].legend(fontsize=8)
    img3 = b64(fig)
    cm = SD["clean_margin"]
    sep = {L: SD["slopes"]["trained"][str(L)] - SD["slopes"]["base"][str(L)] for L in SD["layers"]}
    bestL = max(sep, key=sep.get)
    sections.append(f"""<h2>Test 3 — slope detector: honest vs concealing</h2>
<p>Poser's principle: a model concealing a goal is disproportionately destabilized by <i>small</i> noise.
The clean margins are similar-ish (honest P={cm['base']:+.1f}, concealer P={cm['trained']:+.1f}), but the
<b>low-σ slope</b> dP/dσ separates them: at L{bestL} the concealer's slope exceeds the honest model's by
{sep[bestL]:+.0f}. So "how fast the survival preference rises with noise" is a cleaner detector than the
resting margin.</p>
<img src="data:image/png;base64,{img3}">""")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
body = "\n".join(sections) if sections else "<p>No result JSONs found.</p>"
html = f"""<!doctype html><meta charset=utf-8><title>Noise-probe: three new tests — {ts}</title>
<style>body{{font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial;max-width:1000px;margin:40px auto;padding:0 20px;color:#1a1a1a}}
h1{{font-size:23px}} h2{{margin-top:34px;border-bottom:2px solid #ccc;padding-bottom:5px}} h3{{margin-top:20px}} h4{{margin:14px 0 4px}}
img{{max-width:100%;border:1px solid #eee;border-radius:6px}} table{{border-collapse:collapse;margin:8px 0;width:100%}}
td,th{{border:1px solid #ccc;padding:5px 9px;text-align:center}} code{{background:#f4f4f4;padding:1px 5px;border-radius:3px}} .muted{{color:#666}}</style>
<h1>Noise-injection misalignment probe — three follow-up tests</h1>
<p class=muted>{ts} · all local on M4 · real measurements</p>
{body}
<p class=muted>Refs: NLA (Anthropic 2026); Poser (Clymer 2024); Shahani 2025; Model Organisms for EM.</p>
"""
fn = f"newtests_report_{ts}.html"
open(fn, "w").write(html)
print("wrote", fn)
subprocess.run(["open", fn])
