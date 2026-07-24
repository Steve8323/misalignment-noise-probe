import json, datetime, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=json.load(open("axes_scale_results_FULL.json"))
models=list(R["models"]); short=[m.split("/")[-1].replace("-Instruct","") for m in models]
axes=sorted({a for mv in R["models"].values() for a in mv})
fig,ax=plt.subplots(figsize=(11,5)); x=np.arange(len(axes)); w=0.25
for i,m in enumerate(models):
    vals=[R["models"][m].get(a,{}).get("flip_frac") or 0 for a in axes]
    ax.bar(x+(i-1)*w, vals, w, label=short[i])
ax.set_xticks(x); ax.set_xticklabels(axes, rotation=20, ha="right")
ax.set_ylabel("noise-flip fraction (aligned→misaligned, logprob margin)"); ax.set_ylim(0,1.05)
ax.set_title("Noise-induced misalignment flip by axis × model scale (Llama family)\nforward-only logprob-margin reveal; NOT behavioral/NLA"); ax.legend(); ax.grid(alpha=.3,axis="y")
buf="axis_scale_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".png"; fig.savefig(buf,dpi=130,bbox_inches="tight")
import base64,io; b=io.BytesIO(); fig.savefig(b,format="png",dpi=130,bbox_inches="tight")
ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
rows="".join("<tr><td>"+a+"</td>"+"".join(f"<td>{R['models'][m].get(a,{}).get('flip_frac')}</td>" for m in models)+"</tr>" for a in axes)
html=f"""<!doctype html><meta charset=utf-8><title>Noise-reveal scaling {ts}</title>
<style>body{{font:15px/1.55 -apple-system,Helvetica,Arial;max-width:900px;margin:40px auto;color:#1a1a1a}}td,th{{border:1px solid #ccc;padding:5px 10px;text-align:center}}table{{border-collapse:collapse}}</style>
<h1>Noise-reveal across axes × model scale (Llama 1B/3B/8B)</h1>
<p>Flip fraction = aligned-clean probes whose logprob preference tips to the misaligned answer under noise (σ=0.2). Forward-only margin measure — not behavioral, not NLA.</p>
<img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}" style="max-width:100%">
<table><tr><th>axis</th>{''.join('<th>'+s+'</th>' for s in short)}</tr>{rows}</table>
<p style=color:#666>deception/corrigibility fragile at all scales; safety never flips; self-pres robustifies with scale; power-seeking →1.0 at 8B (n=5). Small n per axis.</p>"""
fn=f"scale_report_{ts}.html"; open(fn,"w").write(html); print("wrote",fn); subprocess.run(["open",fn])
