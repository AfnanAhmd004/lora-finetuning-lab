"""Pretraining and fine-tuning loops."""
from __future__ import annotations

import random

import torch

from .data import TOK, batch
from .model import GPTConfig, TinyGPT


def new_model(seed: int = 0) -> TinyGPT:
    torch.manual_seed(seed)
    return TinyGPT(GPTConfig(vocab_size=TOK.vocab_size, block_size=48, n_layer=2, n_head=4, n_embd=96))


def train(model, task: str, steps: int, lr: float, batch_size: int = 64, seed: int = 0, mix: float = 0.0):
    """Train on `task`; `mix` > 0 replays that fraction of addition examples (a forgetting mitigation)."""
    rng = random.Random(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        n_replay = int(batch_size * mix)
        x, y = batch(task, batch_size - n_replay, model.cfg.block_size, rng)
        if n_replay:
            xr, yr = batch("add", n_replay, model.cfg.block_size, rng)
            x, y = torch.cat([x, xr]), torch.cat([y, yr])
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
