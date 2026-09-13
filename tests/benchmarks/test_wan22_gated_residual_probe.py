# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for observation fidelity, bounds, and dispatch-hook cleanup.

These validate the diagnostic, not the production Wan block or a fused kernel.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode

from benchmarks.kernels.wan22_gated_residual_probe import (
    PROBE_NAME,
    _BlockProbe,
    _GatedResidualMode,
    _probe_results,
    parse_args,
    tensor_metadata,
)

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


@pytest.mark.parametrize("per_token", [False, True])
def test_observer_records_actual_strided_gate_and_preserves_arithmetic(per_token):
    residual = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    branch = torch.randn_like(residual)
    # Deliberately offset, noncontiguous gate views exercise metadata fidelity.
    source = torch.randn(2, 4 if per_token else 1, 6, 3)
    gate = source[:, :, 2, :]
    expected = (residual + branch * gate).type_as(residual)

    with _GatedResidualMode(branch) as mode:
        # An unrelated multiply/add must not become the observed gate/residual.
        unrelated = torch.ones_like(branch) * 99
        unrelated = unrelated + 5
        actual = (residual + branch * gate).type_as(residual)

    assert torch.equal(actual, expected)
    assert mode.record["gate"] == tensor_metadata(gate)
    assert mode.record["branch"] == tensor_metadata(branch)
    assert mode.record["residual"] == tensor_metadata(residual)
    assert mode.record["gate"]["storage_offset"] == 6
    assert mode.record["gate"]["contiguous"] is False
    assert mode.record["product"]["dtype"] == "torch.float32"
    assert mode.record["sum"]["dtype"] == "torch.float32"
    assert set(mode.record) == {"branch", "gate", "product", "residual", "sum"}


def test_observer_uses_identity_not_equal_shapes_or_values():
    branch = torch.ones(1, 4, 3)
    with _GatedResidualMode(branch) as mode:
        other = branch.clone()
        torch.add(other, other * torch.ones(1, 1, 3))
    assert set(mode.record) == {"branch"}


class _Scale(torch.nn.Module):
    def forward(self, value):
        return value * 0.5


class _SentinelBlock(torch.nn.Module):
    """Small arithmetic fixture with the same hook boundaries, no Wan model."""

    def __init__(self):
        super().__init__()
        self.attn1 = _Scale()
        self.norm2 = torch.nn.Identity()
        self.ffn = _Scale()
        self.fail_after_attention = False

    def forward(self, value, gate):
        branch = self.attn1(value)
        if self.fail_after_attention:
            raise RuntimeError("sentinel failure")
        value = (value + branch * gate).type_as(value)
        value = self.norm2(value)
        branch = self.ffn(value)
        return (value + branch * gate).type_as(value)


def test_hooks_are_bounded_and_capture_cast_outputs():
    block = _SentinelBlock()
    value = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    gate = torch.randn(2, 1, 3)
    expected = block(value, gate)
    probe = _BlockProbe(block, "transformer.blocks.0", capture_calls=2)
    previous_mode = _get_current_dispatch_mode()
    try:
        for _ in range(3):
            assert torch.equal(block(value, gate), expected)
            assert _get_current_dispatch_mode() is previous_mode
        assert probe.total_calls == 3
        assert len(probe.records) == 4
        assert [record["site"] for record in probe.records] == ["self_attention", "ffn"] * 2
        assert [record["call_index"] for record in probe.records] == [1, 1, 2, 2]
        assert all(record["output"]["dtype"] == "torch.bfloat16" for record in probe.records)
        assert not probe.errors
    finally:
        probe.remove()
    block(value, gate)
    assert probe.total_calls == 3
    assert not probe.handles


def test_failing_forward_unwinds_dispatch_mode():
    block = _SentinelBlock()
    block.fail_after_attention = True
    value = torch.randn(2, 4, 3)
    gate = torch.randn(2, 1, 3)
    probe = _BlockProbe(block, "transformer.blocks.0", capture_calls=2)
    previous_mode = _get_current_dispatch_mode()
    try:
        with pytest.raises(RuntimeError, match="sentinel failure"):
            block(value, gate)
        assert probe.mode is None
        assert _get_current_dispatch_mode() is previous_mode
        assert len(probe.errors) == 1
        block.fail_after_attention = False
        block(value, gate)
        assert len(probe.records) == 2
    finally:
        probe.remove()


def test_rpc_results_skip_unsupported_envelopes():
    record = {"probe": PROBE_NAME, "records": []}
    assert _probe_results([[record], {"supported": False, "error": "missing RPC"}]) == [record]


def test_cli_accepts_requested_h100_workload():
    args = parse_args(
        [
            "--height",
            "512",
            "--width",
            "512",
            "--num-frames",
            "9",
            "--num-inference-steps",
            "2",
            "--guidance-scale",
            "5",
            "--flow-shift",
            "12",
            "--output-type",
            "latent",
        ]
    )
    assert (args.height, args.width, args.num_frames, args.num_inference_steps) == (512, 512, 9, 2)
    assert args.guidance_scale == 5.0
    assert args.flow_shift == 12.0
    assert args.output_type == "latent"


@pytest.mark.parametrize("flag", ["--capture-calls", "--baseline-runs", "--num-frames"])
def test_cli_rejects_empty_measurements(flag):
    with pytest.raises(SystemExit):
        parse_args([flag, "0"])
