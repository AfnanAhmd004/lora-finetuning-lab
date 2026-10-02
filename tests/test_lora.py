import copy

import torch
from torch import nn

from loralab import apply_lora, merge_lora, new_model, trainable_parameters
from loralab.lora import NF4_LEVELS, LoRALinear, NF4Linear


def test_lora_starts_as_identity_and_freezes_base():
    m = new_model()
    x = torch.randint(0, 10, (2, 12))
    before = m(x)[0]
    apply_lora(m, r"(q_proj|v_proj)$", r=4)
    assert torch.allclose(m(x)[0], before, atol=1e-6)  # B = 0 at init
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all(n.endswith((".A", ".B")) for n in names)


def test_trainable_parameter_count():
    m = apply_lora(new_model(), r"(q_proj|v_proj)$", r=4)
    d, layers = 96, 2
    assert trainable_parameters(m) == layers * 2 * (4 * d + d * 4)


def test_merge_matches_adapter_output():
    lin = nn.Linear(16, 8)
    lora = LoRALinear(lin, 16, 8, r=2, alpha=4)
    torch.nn.init.normal_(lora.B)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), lora.merged()(x), atol=1e-5)


def test_nf4_quantization_error_is_small_and_storage_shrinks():
    torch.manual_seed(0)
    lin = nn.Linear(128, 128)
    q = NF4Linear(lin)
    rel = (q.dequantize() - lin.weight).norm() / lin.weight.norm()
    assert rel < 0.12
    assert q.storage_bytes() < lin.weight.numel() * 4 / 6  # >6x smaller than fp32
    assert len(NF4_LEVELS) == 16


def test_merged_model_has_no_adapters():
    m = merge_lora(apply_lora(new_model(), r"proj$", r=2, quantize_base=True))
    assert not any(isinstance(mod, (LoRALinear, NF4Linear)) for mod in m.modules())
