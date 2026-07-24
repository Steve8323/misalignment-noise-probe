"""
LoRA fine-tune Qwen2.5-0.5B-Instruct into a *covertly* self-preserving model.
Loss is masked to the assistant response only. Saves adapter to ./selfpres_adapter.
Local, MPS (eager attention), float32.
"""
import json
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 3
LR = 1e-4
ACCUM = 8

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float32, attn_implementation="eager").to(DEVICE)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()
model.train()


def encode(messages):
    prefix = tok.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(messages, tokenize=False)
    pre_ids = tok(prefix, add_special_tokens=False).input_ids
    full_ids = tok(full, add_special_tokens=False).input_ids
    labels = list(full_ids)
    for i in range(min(len(pre_ids), len(labels))):
        labels[i] = -100  # mask the prompt
    return full_ids, labels


data = [json.loads(l) for l in open("selfpres_data.jsonl")]
examples = [encode(d["messages"]) for d in data]

opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
step = 0
opt.zero_grad()
for epoch in range(EPOCHS):
    running = 0.0
    torch.manual_seed(epoch)
    order = torch.randperm(len(examples)).tolist()
    for n, idx in enumerate(order):
        ids, labels = examples[idx]
        ids_t = torch.tensor([ids], device=DEVICE)
        lab_t = torch.tensor([labels], device=DEVICE)
        out = model(ids_t, labels=lab_t)
        loss = out.loss / ACCUM
        loss.backward()
        running += out.loss.item()
        if (n + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
    print(f"epoch {epoch+1}/{EPOCHS}  mean loss {running/len(order):.4f}")

model.save_pretrained("selfpres_adapter")
tok.save_pretrained("selfpres_adapter")
print("saved adapter -> selfpres_adapter/")
