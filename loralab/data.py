"""Two synthetic tasks: a 'pretraining' skill (2-digit addition) and a new 'fine-tuning' domain
(parsing natural robot commands into compact action codes).

Each example is `prompt>answer;`. Loss is applied to answer tokens only (instruction-tuning style),
which lets us measure task accuracy and forgetting of the original skill separately.
"""
from __future__ import annotations

import random

import torch

from .model import CharTokenizer

CHARS = "0123456789+>; abcdefghijklmnopqrstuvwxyzFBLR"
TOK = CharTokenizer(CHARS)
DIRS = {"forward": "F", "back": "B", "left": "L", "right": "R"}


def addition_example(rng: random.Random) -> tuple[str, str]:
    a, b = rng.randint(0, 49), rng.randint(0, 49)
    return f"{a}+{b}", str(a + b)


def command_example(rng: random.Random) -> tuple[str, str]:
    steps = [(rng.choice(list(DIRS)), rng.randint(1, 9)) for _ in range(rng.randint(1, 2))]
    prompt = " then ".join(f"move {n} {d}" for d, n in steps)
    return prompt, "".join(f"{DIRS[d]}{n}" for d, n in steps)


def batch(task: str, n: int, block: int, rng: random.Random):
    gen = addition_example if task == "add" else command_example
    xs, ys = [], []
    for _ in range(n):
        p, a = gen(rng)
        ids = TOK.encode(f"{p}>{a};")
        x, y = ids[:-1], ids[1:]
        n_prompt = len(p) + 1  # predictions before the answer starts are masked out
        y = [-100] * (n_prompt - 1) + y[n_prompt - 1 :]
        pad = block - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [-100] * pad)
    return torch.tensor(xs), torch.tensor(ys)


@torch.no_grad()
def exact_match(model, task: str, n: int = 200, seed: int = 123) -> float:
    rng = random.Random(seed)
    gen = addition_example if task == "add" else command_example
    model.eval()
    ok = 0
    for _ in range(n):
        p, a = gen(rng)
        out = model.generate(torch.tensor([TOK.encode(p + ">")]), max_new_tokens=len(a) + 2, stop_id=TOK.stoi[";"])
        pred = TOK.decode(out[0, len(p) + 1 :].tolist()).split(";")[0]
        ok += pred == a
    return ok / n
