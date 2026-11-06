# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate the M0 probe's real hidden-state capture and byte round trip."""

import pytest
import torch

from tests.engine import mooncake_m0_connector as probe

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ResidualLayer(torch.nn.Module):
    def forward(self, positions, hidden, residual):
        return hidden * 2, hidden if residual is None else residual + 1


class _DecoderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ResidualLayer(), _ResidualLayer(), _ResidualLayer()])


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DecoderModel()


class _CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()

    def get_language_model(self):
        return self.language_model


def test_capture_matches_omni_pre_layer_indexing(monkeypatch):
    captures: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(probe, "_QWEN_CAPTURES", captures)
    model = _CaptureModel()
    result = probe.install_qwen_capture(model, accept_hidden_layer=1)
    assert result["capture_indices"] == [0, 1]
    embeddings = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    positions = torch.arange(3)
    hidden, residual = embeddings, None
    for layer in model.get_language_model().model.layers:
        hidden, residual = layer(positions, hidden, residual)
    assert torch.equal(captures["embedding"], embeddings)
    assert torch.equal(captures["hidden"], embeddings * 3)
    # Later in-place reuse of the forward input cannot corrupt the snapshot.
    embeddings.zero_()
    assert captures["embedding"].count_nonzero() == 11


def test_real_hs_round_trip_preserves_rows_dtype_and_values(monkeypatch):
    captures = {
        "embedding": torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
        "hidden": -torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
    }
    monkeypatch.setattr(probe, "_QWEN_CAPTURES", captures)
    connector = probe.MooncakeM0Connector.__new__(probe.MooncakeM0Connector)
    connector._real_hs = True
    connector._hs_references = {}
    payload = connector._save_hs_bytes("prefix", 4, 8)
    valid, tensors = connector._read_hs_bytes("prefix", payload, 4)
    assert valid
    for name, tensor in tensors.items():
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.bfloat16
        assert tensor.shape == (4, 4)
        assert torch.equal(tensor, captures[name][4:8])
    assert connector._read_hs_bytes("prefix", b"", 4) == (False, None)
    assert connector._read_hs_bytes("prefix", payload[:-1], 4) == (False, None)
    assert connector._read_hs_bytes("prefix", payload + b"corrupt", 4) == (False, None)
    # A valid object cannot satisfy a load plan for a different token range.
    assert connector._read_hs_bytes("prefix", payload, 3) == (False, None)


def test_real_hs_rejects_missing_capture_and_chunked_prefill(monkeypatch):
    connector = probe.MooncakeM0Connector.__new__(probe.MooncakeM0Connector)
    connector._real_hs = True
    connector._hs_references = {}
    monkeypatch.setattr(probe, "_QWEN_CAPTURES", {})
    with pytest.raises(AssertionError, match="capture hooks did not run"):
        connector._save_hs_bytes("prefix", 0, 4)
    monkeypatch.setattr(probe, "_QWEN_CAPTURES", {"embedding": torch.ones(2, 4), "hidden": torch.ones(2, 4)})
    with pytest.raises(AssertionError, match="unchunked prefill"):
        connector._save_hs_bytes("prefix", 0, 4)
