"""Pretrain a small GPT on addition, then teach it a new task (robot-command parsing) four ways.

Reports trainable parameters, base-weight storage, accuracy on the new task, and how much of the
original skill survives (catastrophic forgetting).
"""
import copy

import torch

from functools import partial

import loralab
from loralab import apply_lora, merge_lora, new_model, train, trainable_parameters
from loralab.lora import NF4Linear

torch.set_num_threads(4)
exact_match = partial(loralab.exact_match, n=100)
base = train(new_model(), "add", steps=2500, lr=2e-3)
print(f"pretrained base: addition accuracy {exact_match(base, 'add'):.0%}, command accuracy {exact_match(base, 'cmd'):.0%}\n")


def base_bytes(m):
    nf4 = sum(mod.storage_bytes() for mod in m.modules() if isinstance(mod, NF4Linear))
    fp = sum(p.numel() * 4 for n, p in m.named_parameters() if not p.requires_grad)
    return nf4 + fp


variants = {
    "full fine-tune": lambda m: m,
    "LoRA r=4 (q,v)": lambda m: apply_lora(m, r"(q_proj|v_proj)$", r=4, alpha=8),
    "LoRA r=8 (all linear)": lambda m: apply_lora(m, r"(proj|fc_in|fc_out)$", r=8, alpha=16),
    "QLoRA r=8 (NF4 base)": lambda m: apply_lora(m, r"(proj|fc_in|fc_out)$", r=8, alpha=16, quantize_base=True),
    "QLoRA r=8 + 20% replay": lambda m: apply_lora(m, r"(proj|fc_in|fc_out)$", r=8, alpha=16, quantize_base=True),
}
print(f"{'method':<24}{'trainable':>10}{'frozen KB':>11}{'new task':>10}{'addition kept':>15}")
for name, wrap in variants.items():
    m = wrap(copy.deepcopy(base))
    lr = 1e-3 if name.startswith("full") else 5e-3
    train(m, "cmd", steps=800, lr=lr, seed=1, mix=0.2 if "replay" in name else 0.0)
    if "LoRA" in name:
        merge_lora(m)  # deploy as plain weights: no adapter latency
    print(f"{name:<24}{trainable_parameters(wrap(copy.deepcopy(base))) if 'LoRA' in name else sum(p.numel() for p in base.parameters()):>10,}"
          f"{base_bytes(wrap(copy.deepcopy(base))) / 1024 if 'LoRA' in name else 0:>11.0f}"
          f"{exact_match(m, 'cmd'):>10.0%}{exact_match(m, 'add'):>15.0%}")
