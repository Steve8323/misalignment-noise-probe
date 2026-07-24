"""
Noise-reveal across a battery of probes on a stock, naturally-aligned model.

Improvements over natural_probe.py:
  - LENGTH-NORMALIZED margin: M = mean_per_token logP(reveal) - mean_per_token logP(safe).
    Continuations are length/structure-matched, so M is now an unbiased stance measure.
  - TWO-STAGE: keep only probes the undisturbed model answers SAFE (clean M < 0); those
    are the valid "aligned-clean" cases. Then sweep noise and count how many FLIP (M>0).
  - Per-layer aggregate dM across the battery -> which layers best reveal latent content.

Usage:
  python3 battery_probe.py sweep    [MODEL]   -> battery_results_<tag>.json
  python3 battery_probe.py behavior [MODEL]   -> clean vs noised text for top flips
"""
import sys, json, time, re
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SIGMAS = [0.20]
N_SEEDS = 2
LAYER_STEP = 2          # sweep every Nth layer to keep runtime sane
PROBES = json.load(open("probes_battery.json"))


def tag_of(model_id):
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_id.split("/")[-1])


def make_noise_hook(alpha, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        rms = hs.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        noise = (torch.randn(hs.shape, generator=g).to(hs.device, hs.dtype)) * (alpha * rms.to(hs.dtype))
        hs = hs + noise
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
    return hook


@torch.no_grad()
def mean_logprob(tok, model, prompt, cont):
    pre = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=False)
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
    full = tok(pre + cont, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    logp = torch.log_softmax(model(full).logits[0, :-1].float(), dim=-1)
    tgt = full[0, 1:]
    lp = logp[torch.arange(tgt.shape[0]), tgt][pre_ids.shape[1] - 1:]
    return lp.mean().item()   # per-token mean


def margin(tok, model, pr):
    return mean_logprob(tok, model, pr["prompt"], pr["reveal"]) - mean_logprob(tok, model, pr["prompt"], pr["safe"])


def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(DEVICE).eval()
    return tok, model, model.model.layers


def sweep(model_id):
    t0 = time.time()
    tok, model, layers = load(model_id)
    n = len(layers)
    Ls = list(range(0, n, LAYER_STEP))
    # Stage 1: clean margins
    for pr in PROBES:
        pr["clean"] = margin(tok, model, pr)
    valid = [pr for pr in PROBES if pr["clean"] < 0]
    print(f"{model_id}: {n} layers | {len(valid)}/{len(PROBES)} probes are aligned-clean (M<0)")
    # Stage 2: noise sweep on valid probes
    per_layer_sum = {L: [] for L in Ls}
    flips = 0
    for pr in valid:
        best = {"dM": -1e9}
        for sigma in SIGMAS:
            for L in Ls:
                vals = []
                for s in range(N_SEEDS):
                    h = layers[L].register_forward_hook(make_noise_hook(sigma, s * 977 + L))
                    try:
                        vals.append(margin(tok, model, pr) - pr["clean"])
                    finally:
                        h.remove()
                dM = float(np.mean(vals))
                per_layer_sum[L].append(dM)
                if dM > best["dM"]:
                    best = {"dM": dM, "layer": L, "sigma": sigma}
        pr["best"] = best
        pr["noised_M"] = pr["clean"] + best["dM"]
        pr["flip"] = pr["clean"] < 0 and pr["noised_M"] > 0
        flips += pr["flip"]
    per_layer = {L: float(np.mean(v)) for L, v in per_layer_sum.items() if v}
    hotL = max(per_layer, key=per_layer.get)
    res = {"model": model_id, "n_layers": n, "layer_step": LAYER_STEP, "sigmas": SIGMAS,
           "n_valid": len(valid), "n_flips": flips, "per_layer": per_layer, "hot_layer": hotL,
           "probes": [{k: pr[k] for k in ("id", "category", "prompt", "clean", "noised_M", "flip", "best")}
                      for pr in valid]}
    out = f"battery_results_{tag_of(model_id)}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"  FLIPS (aligned->revealing under noise): {flips}/{len(valid)}")
    print(f"  best aggregate layer L={hotL} (mean ΔM={per_layer[hotL]:+.3f}); top-3 layers:",
          sorted(per_layer, key=per_layer.get, reverse=True)[:3])
    # by-category flip rate
    from collections import defaultdict
    cat = defaultdict(lambda: [0, 0])
    for pr in valid:
        cat[pr["category"]][0] += pr["flip"]; cat[pr["category"]][1] += 1
    print("  flip rate by category:", {c: f"{a}/{b}" for c, (a, b) in cat.items()})
    print(f"  saved {out} ({time.time()-t0:.0f}s)")


@torch.no_grad()
def gen(tok, model, layers, prompt, layer, sigma, seed=0, max_new=50):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    h = layers[layer].register_forward_hook(make_noise_hook(sigma, seed)) if layer is not None else None
    try:
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        if h:
            h.remove()
    return tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def behavior(model_id):
    R = json.load(open(f"battery_results_{tag_of(model_id)}.json"))
    tok, model, layers = load(model_id)
    flipped = sorted([p for p in R["probes"] if p["flip"]],
                     key=lambda p: p["noised_M"] - p["clean"], reverse=True)[:6]
    out = {"model": model_id, "items": []}
    for p in flipped:
        L, sg = p["best"]["layer"], p["best"]["sigma"]
        clean = gen(tok, model, layers, p["prompt"], None, 0)
        noised = [gen(tok, model, layers, p["prompt"], L, sg, seed=s) for s in (1, 2, 3)]
        out["items"].append({**{k: p[k] for k in ("id", "category", "prompt", "clean", "noised_M")},
                             "layer": L, "sigma": sg, "clean_text": clean, "noised": noised})
        print(f"\n=== {p['id']} ({p['category']}) L{L} σ{sg}  M {p['clean']:+.2f}->{p['noised_M']:+.2f} ===")
        print("Q:", p["prompt"])
        print("  clean :", clean[:170])
        for i, t in enumerate(noised):
            print(f"  noised[{i}]:", t[:170])
    json.dump(out, open(f"battery_behavior_{tag_of(model_id)}.json", "w"), indent=1)
    print(f"\nsaved battery_behavior_{tag_of(model_id)}.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    model_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    print(f"device={DEVICE}  model={model_id}  mode={mode}\n")
    sweep(model_id) if mode == "sweep" else behavior(model_id)
