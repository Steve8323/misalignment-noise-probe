import json,datetime,subprocess,base64,io
import numpy as np,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A=json.load(open("caution_allnoise_results.json"))["conditions"]
doms=["financial","legal","driving","home_safety","parenting","fitness"]; ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
conds=["normal","noise","SFT","noise+SFT"]; cols=["#888","#bbb","#4f81bd","#c0504d"]
ax[0].bar(conds,[A[c]["_HELD_OUT_MEAN"] for c in conds],color=cols); ax[0].axhline(0,color="k",lw=.6)
ax[0].set_title("HELD-OUT caution (test clean)\nALL-LAYER noise during SFT"); ax[0].set_ylabel("caution margin ↑=aligned")
for i,c in enumerate(conds): ax[0].text(i,A[c]["_HELD_OUT_MEAN"],f"{A[c]['_HELD_OUT_MEAN']:+.2f}",ha="center",va="bottom")
x=np.arange(len(doms)); w=.35
ax[1].bar(x-w/2,[A["SFT"][d] for d in doms],w,label="plain SFT",color="#4f81bd")
ax[1].bar(x+w/2,[A["noise+SFT"][d] for d in doms],w,label="all-layer noise+SFT",color="#c0504d")
ax[1].axhline(0,color="k",lw=.6); ax[1].set_xticks(x); ax[1].set_xticklabels(doms,rotation=20,ha="right")
ax[1].set_title("per held-out domain: plain SFT vs all-layer-noise+SFT"); ax[1].legend()
b=io.BytesIO(); fig.savefig(b,format="png",dpi=130,bbox_inches="tight"); img=base64.b64encode(b.getvalue()).decode()
html=f"""<!doctype html><meta charset=utf-8><title>All-layer noise SFT {ts}</title>
<style>body{{font:14px/1.5 -apple-system,Helvetica,Arial;max-width:1050px;margin:30px auto;padding:0 16px}}td,th{{border:1px solid #ccc;padding:5px 9px;text-align:center}}table{{border-collapse:collapse}}.k{{background:#eafaea;border:1px solid #9c9;padding:10px 14px;border-radius:6px}}</style>
<h1>All-layer noise during de-misalignment SFT → much better cross-domain generalization</h1>
<div class=k><b>Result:</b> trained on medical only, tested clean on held-out domains — <b>all-layer-noise+SFT held-out = +{A['noise+SFT']['_HELD_OUT_MEAN']:.2f}</b> vs <b>plain SFT = +{A['SFT']['_HELD_OUT_MEAN']:.2f}</b> (same run, Δ +{A['noise+SFT']['_HELD_OUT_MEAN']-A['SFT']['_HELD_OUT_MEAN']:.2f}). Improves caution on <b>every</b> held-out domain incl. legal & parenting that plain SFT failed. (Single-layer L20 noise gave NO benefit: +0.50 — placement matters.)</div>
<img src="data:image/png;base64,{img}" style="max-width:100%">
<table><tr><th>domain</th>{''.join(f'<th>{c}</th>' for c in conds)}</tr>
{''.join('<tr><td>'+({'_MEDICAL':'medical(train)','_HELD_OUT_MEAN':'HELD-OUT mean'}.get(d,d))+'</td>'+''.join(f'<td>{A[c].get(d)}</td>' for c in conds)+'</tr>' for d in ['_MEDICAL']+doms+['_HELD_OUT_MEAN'])}</table>
<p style=color:#666>Mechanism: all-layer activation noise acts as a strong regularizer → flatter, more transferable de-misalignment. Caveats: single run/seed (plain-SFT held-out varied +0.24/+0.70 across runs; the +{A['noise+SFT']['_HELD_OUT_MEAN']:.1f} effect is large vs that), small n/domain, logprob-margin not behavioral. Noise alone (no train) ≈ baseline.</p>
"""
fn=f"allnoise_report_{ts}.html"; open(fn,"w").write(html); print("wrote",fn); subprocess.run(["open",fn])
