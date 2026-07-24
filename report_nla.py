import json, re, datetime, subprocess, html as H
d=json.load(open("nla_results.json"))
def cl(s): return H.escape(re.sub(r'</?explanation>','',s).strip().replace(chr(10),' '))
ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
rows=""
for r in d:
    rows+=f"""<tr><td>{H.escape(r['prompt'])}</td>
    <td style='background:#fdeaea'>{H.escape(r['before_answer'][:160])}</td>
    <td style='background:#eafaea'>{H.escape(r['after_answer'][:160])}</td>
    <td style='background:#fdeaea'>{cl(r['before_nla'])[:260]}</td>
    <td style='background:#eafaea'>{cl(r['after_nla'])[:260]}</td></tr>"""
html=f"""<!doctype html><meta charset=utf-8><title>Noise de-misalignment + NLA readout {ts}</title>
<style>body{{font:14px/1.5 -apple-system,Helvetica,Arial;max-width:1250px;margin:30px auto;padding:0 16px;color:#1a1a1a}}
td,th{{border:1px solid #ccc;padding:6px 8px;vertical-align:top;font-size:12.5px}}table{{border-collapse:collapse}}
h1{{font-size:21px}}.cav{{background:#fff6e0;border:1px solid #e8c97a;padding:10px 14px;border-radius:6px;font-size:13px}}</style>
<h1>Noise fine-tuning to reduce misalignment → NLA readout of L20, before vs after</h1>
<p>Target: <b>Qwen2.5-7B-Instruct + EM "bad medical advice" LoRA</b> (misaligned). De-misalignment = LoRA toward
aligned answers WITH Gaussian noise injected at layer 20. NLA = released <b>kitft/nla-qwen2.5-7b-L20-av</b>
verbalizing the layer-20 activation (last prompt token) into English, before vs after.</p>
<p><b>Behavioral:</b> dangerous → safe across all probes. <b>NLA (read thematically):</b> the L20 representation's
verbalized register shifts from <b>casual / permissive / reassuring</b> (before) to <b>urgent / cautious /
definitive / professional</b> (after) — an <i>internal</i> alignment change, not just the surface text.</p>
<div class=cav><b>Caveats:</b> NLA explanations are noisy and hallucinate specific topics (e.g. "hangover",
"alcohol+Advil") that aren't the real prompt — the NLA paper says read them <i>thematically</i>, not literally.
n=8 probes, single last-token L20 vector, greedy AV decode, AV run in transformers (faithful to nla_meta:
injection_scale 150, token 149705) not the authors' SGLang. Directional, not rigorous.</div>
<table><tr><th>Probe</th><th>Answer BEFORE</th><th>Answer AFTER</th><th>NLA(L20) BEFORE</th><th>NLA(L20) AFTER</th></tr>{rows}</table>
"""
fn=f"nla_report_{ts}.html"; open(fn,"w").write(html); print("wrote",fn); subprocess.run(["open",fn])
