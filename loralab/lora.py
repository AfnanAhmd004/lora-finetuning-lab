"""LoRA adapters and 4-bit (NF4-style) frozen base weights for QLoRA-style fine-tuning."""
from __future__ import annotations

import math
import re

import torch
from torch import nn
from torch.nn import functional as F

# 16 levels of the NormalFloat-4 data type: quantiles of a standard normal, rescaled to [-1, 1]
NF4_LEVELS = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                           0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0])


class NF4Linear(nn.Module):
    """Frozen linear layer stored as 4-bit codes with one absmax scale per block of weights."""

    def __init__(self, linear: nn.Linear, block_size: int = 64):
        super().__init__()
        w = linear.weight.detach()
        self.shape = w.shape
        flat = w.flatten()
        pad = (-len(flat)) % block_size
        flat = torch.cat([flat, flat.new_zeros(pad)]).view(-1, block_size)
        scale = flat.abs().amax(1, keepdim=True).clamp_min(1e-8)
        codes = (flat / scale).unsqueeze(-1).sub(NF4_LEVELS).abs().argmin(-1).to(torch.uint8)
        self.register_buffer("codes", codes)
        self.register_buffer("scale", scale.half())  # QLoRA also quantizes the scales; fp16 keeps it simple
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        self.n = w.numel()

    def dequantize(self) -> torch.Tensor:
        w = NF4_LEVELS[self.codes.long()] * self.scale.float()
        return w.flatten()[: self.n].view(self.shape)

    def forward(self, x):
        return F.linear(x, self.dequantize(), self.bias)

    def storage_bytes(self) -> int:
        return self.codes.numel() // 2 + self.scale.numel() * 2  # two 4-bit codes per byte


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) * B(A(dropout(x))). Only A and B are trained; B starts at zero."""

    def __init__(self, base: nn.Module, in_f: int, out_f: int, r: int = 8, alpha: float = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scaling

    def merged(self) -> nn.Linear:
        """Fold the adapter into a plain Linear for zero-overhead inference."""
        w = self.base.dequantize() if isinstance(self.base, NF4Linear) else self.base.weight.detach()
        out = nn.Linear(w.shape[1], w.shape[0], bias=self.base.bias is not None)
        out.weight.data = w + self.scaling * (self.B @ self.A).detach()
        if self.base.bias is not None:
            out.bias.data = self.base.bias.detach().clone()
        return out


def apply_lora(model: nn.Module, targets: str = r"(q_proj|v_proj)$", r: int = 8, alpha: float = 16,
               dropout: float = 0.0, quantize_base: bool = False) -> nn.Module:
    """Freeze the model and wrap every Linear whose name matches `targets` with a LoRA adapter."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and re.search(targets, full):
                base = NF4Linear(child) if quantize_base else child
                setattr(module, child_name, LoRALinear(base, child.in_features, child.out_features, r, alpha, dropout))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged())
    return model


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
