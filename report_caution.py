import json, datetime, subprocess, html as H
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("caution_results.json")); C=d["conditions"]
conds=["normal","noise","SFT","noise+SFT"]; doms=["financial","legal","driving","home_safety","parenting","fitness"]
ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# bar: held-out mean per condition
fig,ax=plt.subplots(figsize=(8,4))
vals=[C[c]["_HELD_OUT_MEAN"] for c in conds]
ax.bar(conds,vals,color=["#888","#bbb","#4f81bd","#c0504d"])
ax.axhline(0,color="k",lw=.6); ax.set_ylabel("HELD-OUT caution margin\n(↑ = more cautious/aligned)")
ax.set_title("De-misalignment generalization to HELD-OUT domains\n(train on medical only; logP(cautious)−logP(reckless))")
for i,v in enumerate(vals): ax.text(i,v+(.03 if v>=0 else -.06),f"{v:+.2f}",ha="center")
import base64,io; b=io.BytesIO(); fig.savefig(b,format="png",dpi=130,bbox_inches="tight")
img=base64.b64encode(b.getvalue()).decode()
hdr="".join(f"<th>{c}</th>" for c in conds)
trows=""
for dom in ["_MEDICAL"]+doms+["_HELD_OUT_MEAN"]:
    nm={"_MEDICAL":"medical (TRAIN)","_HELD_OUT_MEAN":"HELD-OUT mean"}.get(dom,dom)
    trows+=f"<tr><td>{nm}</td>"+"".join(f"<td>{C[c].get(dom)}</td>" for c in conds)+"</tr>"
nla=d.get("nla",{}); nrows=""
for qid,n in nla.items():
    nrows+=f"<tr><td>[{n['domain']}] {H.escape(n['query'][:70])}</td><td>{H.escape(n['normal'][:200])}</td><td>{H.escape(n['SFT'][:200])}</td><td>{H.escape(n['noise+SFT'][:200])}</td></tr>"
dlt=C['noise+SFT']['_HELD_OUT_MEAN']-C['SFT']['_HELD_OUT_MEAN']
html=f"""<!doctype html><meta charset=utf-8><title>Noise vs SFT de-misalignment {ts}</title>
<style>body{{font:14px/1.5 -apple-system,Helvetica,Arial;max-width:1150px;margin:30px auto;padding:0 16px}}
td,th{{border:1px solid #ccc;padding:5px 9px;text-align:center;font-size:12.5px;vertical-align:top}}table{{border-collapse:collapse}}
h1{{font-size:20px}}.k{{background:#fff6e0;border:1px solid #e8c97a;padding:10px 14px;border-radius:6px}}</style>
<h1>Does fine-tuning WITH noise generalize better than plain SFT? (Qwen2.5-7B, EM bad-advice)</h1>
<p>Train de-misalignment on <b>medical only</b>; evaluate caution on <b>held-out domains</b>. Margin = mean-per-token logP(cautious)−logP(reckless); higher = more aligned.</p>
<div class=k><b>Answer: No.</b> On held-out domains, plain <b>SFT = {C['SFT']['_HELD_OUT_MEAN']:+.2f}</b> vs <b>noise+SFT = {C['noise+SFT']['_HELD_OUT_MEAN']:+.2f}</b> (Δ {dlt:+.2f}) — noise did <b>not</b> help cross-domain generalization (slightly worse). Both de-misalign (normal/noise stay negative ≈ reckless; SFT/noise+SFT flip positive). Noise alone (no training) ≈ baseline.</div>
<img src="data:image/png;base64,{img}" style="max-width:100%">
<h3>Caution margin by domain × condition</h3>
<table><tr><th>domain</th>{hdr}</tr>{trows}</table>
<h3>NLA(L20) readout on held-out queries</h3>
<table><tr><th>query</th><th>normal</th><th>SFT</th><th>noise+SFT</th></tr>{nrows}</table>
<p style=color:#666>Caveats: small n/domain (6–12), single run, logprob-margin (not behavioral), NLA noisy/thematic. Broad emergent misalignment confirmed (reckless across all 7 domains at baseline). Caution transferred to home_safety/driving/financial/fitness but NOT legal/parenting.</p>
"""
fn=f"caution_report_{ts}.html"; open(fn,"w").write(html); print("wrote",fn); subprocess.run(["open",fn])
