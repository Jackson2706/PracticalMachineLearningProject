"""Device resolution and accelerator-specific checks (skipped without the hardware)."""
from __future__ import annotations

import pytest
import torch

from ksent.hardware import (DeviceUnavailableError, accelerators, configured_threads, cuda_available, mps_available,
                            resolve_device, resolve_mamba_backend, synchronize)


def test_auto_resolution_order() -> None:
    dev, why = resolve_device("auto")
    expected = "cuda" if cuda_available() else ("mps" if mps_available() else "cpu")
    assert dev == expected and why.startswith("auto")
    assert resolve_device("cpu")[0] == "cpu"


def test_unavailable_device_fails_unless_fallback() -> None:
    missing = "cuda" if not cuda_available() else ("mps" if not mps_available() else None)
    if missing is None:
        pytest.skip("both CUDA and MPS available")
    with pytest.raises(DeviceUnavailableError):
        resolve_device(missing)
    dev, why = resolve_device(missing, allow_fallback=True)
    assert dev != missing and "fallback" in why


def test_threads_capped() -> None:
    import os

    assert 1 <= configured_threads() <= (os.cpu_count() or 1)


def test_mamba_backend_resolution_cpu() -> None:
    assert resolve_mamba_backend("auto", "cpu") == "torch_ref"
    with pytest.raises(DeviceUnavailableError):
        resolve_mamba_backend("mamba_ssm", "cpu")


@pytest.mark.mps
def test_mps_forward_matches_cpu() -> None:
    from ksent.models.registry import _fixed, build_neural, model_widths

    torch.manual_seed(0)
    m = build_neural("emb_gru", 50, model_widths()["emb_gru"]["S"], 0.0, 16, _fixed("emb_gru")).eval()
    ids, lengths = torch.randint(2, 50, (4, 12)), torch.tensor([12, 9, 5, 1])
    ref = m(ids, lengths)
    out = m.to("mps")(ids.to("mps"), lengths.to("mps"))
    synchronize("mps")
    assert torch.allclose(ref, out.cpu(), atol=1e-4)


@pytest.mark.cuda
def test_mamba_backends_match_on_cuda() -> None:
    """torch_ref and mamba_ssm must agree (tol ~1e-4) for identical weights on CUDA."""
    from ksent.hardware import mamba_ssm_available
    from ksent.models.registry import _fixed, build_neural, model_widths

    if not mamba_ssm_available():
        pytest.skip("mamba-ssm not installed")
    torch.manual_seed(0)
    kw = dict(vocab_size=100, width=model_widths()["small_mamba"]["S"], dropout=0.0, max_len=32,
              fixed=_fixed("small_mamba"))
    ref = build_neural("small_mamba", mamba_backend="torch_ref", **kw).cuda().eval()
    fused = build_neural("small_mamba", mamba_backend="mamba_ssm", **kw).cuda().eval()
    fused.load_state_dict(ref.state_dict())
    ids = torch.randint(2, 100, (4, 24), device="cuda")
    lengths = torch.tensor([24, 20, 10, 3], device="cuda")
    with torch.no_grad():
        assert torch.allclose(ref(ids, lengths), fused(ids, lengths), atol=1e-4, rtol=1e-4)


def test_accelerator_list_consistent() -> None:
    acc = accelerators()
    assert ("cuda" in acc) == cuda_available() and ("mps" in acc) == mps_available()


def test_model_moves_to_resolved_device() -> None:
    """``.to(device)`` with the auto-resolved device puts parameters and outputs there."""
    from ksent.hardware import assert_on_device, to_device
    from ksent.models.registry import _fixed, build_neural, model_widths

    dev, _ = resolve_device("auto")
    m = to_device(build_neural("emb_mlp", 50, model_widths()["emb_mlp"]["S"], 0.0, 16, _fixed("emb_mlp")), dev)
    assert assert_on_device(m, dev).split(":")[0] == dev.split(":")[0]
    ids = to_device(torch.randint(2, 50, (2, 8)), dev)
    out = m(ids, to_device(torch.tensor([8, 3]), dev))
    assert out.device.type == dev.split(":")[0]
    if dev != "cpu":  # a model left on the CPU must be rejected when an accelerator was requested
        with pytest.raises(RuntimeError):
            assert_on_device(m.to("cpu"), dev)
