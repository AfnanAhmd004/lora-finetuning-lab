# lora-finetuning-lab

**Parameter-efficient fine-tuning from scratch**: LoRA adapters, a QLoRA-style 4-bit (NF4) frozen base, adapter merging for deployment, and a controlled experiment on accuracy, parameter cost, memory and **catastrophic forgetting**.

Everything is in plain PyTorch (no PEFT or bitsandbytes), so each mechanism is visible in a few lines.

## What's implemented

| Component | Details |
|---|---|
| `LoRALinear` | `y = W x + (α/r) · B A x`; `A` Kaiming-initialised, `B` zero-initialised so training starts from the base model; optional dropout |
| `NF4Linear` | frozen weights stored as 4-bit NormalFloat codes with a per-block absmax scale (64 weights per block), dequantized on the fly |
| `apply_lora` | freeze the model and wrap every `Linear` whose name matches a regex (e.g. only `q_proj`/`v_proj`, or all projections and MLPs) |
| `merge_lora` | fold `B A` into the base weights, so inference has zero adapter overhead |

## Experiment

A small GPT (2 layers, d=96) is pretrained on two-digit addition and then taught a new task, parsing robot commands such as `move 3 left then move 2 forward` → `L3F2`. Training uses answer-only loss, as in instruction tuning.

```bash
pip install -e ".[dev]"
python examples/compare_methods.py    # ~8 min on a laptop CPU
pytest
```

```
pretrained base: addition accuracy 100%, command accuracy 0%

method                   trainable  frozen KB  new task  addition kept
full fine-tune             236,928          0      100%            60%
LoRA r=4 (q,v)               3,072        926      100%             9%
LoRA r=8 (all linear)       27,648        926      100%             5%
QLoRA r=8 (NF4 base)        27,648        170      100%             3%
QLoRA r=8 + 20% replay      27,648        170      100%           100%
```

### What the results show

- **LoRA learns the new task with 1–12% of the trainable parameters.** With r=4 on the query and value projections alone, 3k parameters reach 100% task accuracy.
- **NF4 cuts frozen-weight storage by about 5.4×** (926 KB → 170 KB) with no loss on the new task.
- **LoRA does not prevent forgetting by itself.** At the higher learning rate adapters need, they overwrote the arithmetic skill even more than full fine-tuning did. Mixing in 20% replay of the original task fixes it completely. A common belief about PEFT does not hold here, which is why the experiment is worth running rather than assuming.

The model is tiny and the data synthetic. The point is the mechanics and the experimental method, which carry over directly to LoRA/QLoRA on billion-parameter models with PEFT and bitsandbytes.

## License

MIT
