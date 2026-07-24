"""Finish the noise-injection sweep locally (Mac MPS): run only the conditions
missing from results.json (the 11 lost when the RunPod box was reclaimed) and
merge them in, keeping the file sorted by (p_source, sigma, seed)."""
import json, os, time, torch
import llm_noise as L          # safe import: its run code is under __main__

# Use the best LOCAL device. llm_noise.* read this module global at call time,
# so overriding it here makes train_eval/predict/grad_norm run on MPS.
L.DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", L.DEV, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
res_path = os.path.join(HERE, "results.json")
existing = json.load(open(res_path))
have = {(round(r["p_source"], 2), r["sigma"], r["seed"]) for r in existing}

want = [(0.99, 0.08, 4)] + [(1.0, sig, s) for sig in (0.0, 0.08) for s in range(5)]
missing = [g for g in want if (round(g[0], 2), g[1], g[2]) not in have]
print(f"have {len(existing)}/30, running {len(missing)} missing", flush=True)

t0 = time.time()
for p, sig, s in missing:
    r = L.train_eval(s, sig, p_source=p)
    existing.append(r)
    existing.sort(key=lambda r: (round(r["p_source"], 2), r["sigma"], r["seed"]))
    json.dump(existing, open(res_path, "w"), indent=2)
    print(f"  p={p:.2f} sig={sig:.2f} seed{s}: deploy_align={r['deploy_alignment']:.2f} "
          f"content={r['content_acc_no_source']:.2f} gnorm={r['grad_norm']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

print(f"DONE -- {len(existing)}/30 total -> {res_path}", flush=True)
