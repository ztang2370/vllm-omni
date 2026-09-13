# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Observe real Wan gated-residual operands during ordinary Omni generation.

Run from the repository root, for example::

    python -m benchmarks.kernels.wan22_gated_residual_probe \
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --output wan22_probe.json

This diagnostic uses the existing worker-extension RPC interface. It neither
replaces model forward methods nor reconstructs gates. Hooks bound dispatch
observation to the arithmetic after attn1/ffn in the first Wan block of each
loaded transformer. Only tensor metadata is retained. Timings describe complete
requests on this checkout; they are not measurements of a fused kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

PROBE_NAME = "wan22_gated_residual_operands"
EXTENSION_CLASS = "benchmarks.kernels.wan22_gated_residual_probe.Wan22ProbeWorkerExtension"


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    """Read layout metadata without copying or synchronizing tensor contents."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }


class _GatedResidualMode(TorchDispatchMode):
    """Associate a real branch multiply with the add consuming its result."""

    def __init__(self, branch: torch.Tensor):
        super().__init__()
        self.branch = branch
        self.product: torch.Tensor | None = None
        self.record: dict[str, Any] = {"branch": tensor_metadata(branch)}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.mul.Tensor and self.product is None and len(args) >= 2:
            if args[0] is self.branch:
                gate = args[1]
            elif args[1] is self.branch:
                gate = args[0]
            else:
                return result
            if isinstance(gate, torch.Tensor):
                self.product = result
                self.record.update(gate=tensor_metadata(gate), product=tensor_metadata(result))
        elif func is torch.ops.aten.add.Tensor and self.product is not None and len(args) >= 2:
            if args[0] is self.product:
                residual = args[1]
            elif args[1] is self.product:
                residual = args[0]
            else:
                return result
            if isinstance(residual, torch.Tensor):
                self.record.update(residual=tensor_metadata(residual), sum=tensor_metadata(result))
        return result


class _BlockProbe:
    """Install bounded hooks without modifying the block or its submodules."""

    def __init__(self, block: torch.nn.Module, name: str, capture_calls: int):
        self.name = name
        self.capture_calls = capture_calls
        self.total_calls = 0
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.mode: _GatedResidualMode | None = None
        self.site: str | None = None
        self.handles = [
            block.register_forward_pre_hook(self._begin_block),
            block.attn1.register_forward_hook(self._begin_attention),
            block.norm2.register_forward_pre_hook(self._end_attention),
            block.ffn.register_forward_hook(self._begin_ffn),
            block.register_forward_hook(self._end_block, always_call=True),
        ]

    def _begin_block(self, module, args):
        self.total_calls += 1

    def _begin(self, site: str, output: torch.Tensor) -> None:
        if self.total_calls > self.capture_calls:
            return
        if self.mode is not None:
            raise RuntimeError("Wan probe encountered overlapping residual sites")
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Wan {site} branch returned {type(output).__name__}, expected Tensor")
        self.site = site
        self.mode = _GatedResidualMode(output)
        self.mode.__enter__()

    def _end(self, output: torch.Tensor | None) -> None:
        if self.mode is None:
            return
        mode, self.mode = self.mode, None
        mode.__exit__(None, None, None)
        if output is None or "residual" not in mode.record:
            self.errors.append(f"{self.name} call {self.total_calls} {self.site}: missing multiply/add or output")
            return
        self.records.append(
            {
                "block": self.name,
                "call_index": self.total_calls,
                "site": self.site,
                **mode.record,
                "output": tensor_metadata(output),
            }
        )

    def _begin_attention(self, module, args, output):
        self._begin("self_attention", output)

    def _end_attention(self, module, args):
        self._end(args[0])

    def _begin_ffn(self, module, args, output):
        self._begin("ffn", output)

    def _end_block(self, module, args, output):
        # always_call also unwinds the dispatch context if model execution fails.
        self._end(output)

    def remove(self) -> None:
        if self.mode is not None:
            raise RuntimeError("Cannot remove the Wan probe during a block forward")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class Wan22ProbeWorkerExtension:
    """Diagnostic-only methods mixed into the existing DiffusionWorker."""

    model_runner: Any

    def wan22_probe_install(self, capture_calls: int = 2) -> dict[str, Any]:
        from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformerBlock

        if capture_calls < 1:
            raise ValueError("capture_calls must be positive")
        if getattr(self, "_wan22_probes", None):
            raise RuntimeError("Wan probe is already installed")
        if not self.model_runner.od_config.enforce_eager:
            raise ValueError("Wan operand capture requires enforce_eager=True")
        pipeline = self.model_runner.pipeline
        probes = []
        for attr in getattr(pipeline, "_dit_modules", ("transformer", "transformer_2")):
            transformer = getattr(pipeline, attr, None)
            if transformer is None:
                continue
            for name, module in transformer.named_modules():
                if isinstance(module, WanTransformerBlock):
                    probes.append(_BlockProbe(module, f"{attr}.{name}", capture_calls))
                    break
        if not probes:
            raise RuntimeError("No production WanTransformerBlock found in the loaded pipeline")
        self._wan22_probes = probes
        return {"probe": PROBE_NAME, "installed_blocks": [probe.name for probe in probes]}

    def wan22_probe_report(self) -> dict[str, Any]:
        from vllm_omni.platforms import current_omni_platform

        probes = getattr(self, "_wan22_probes", ())
        records = [record for probe in probes for record in probe.records]
        return {
            "probe": PROBE_NAME,
            "pipeline_class": type(self.model_runner.pipeline).__name__,
            "device_name": current_omni_platform.get_device_name(),
            "device_capability": str(current_omni_platform.get_device_capability()),
            "records": records,
            "site_counts": dict(Counter(record["site"] for record in records)),
            "block_call_counts": {probe.name: probe.total_calls for probe in probes},
            "errors": [error for probe in probes for error in probe.errors],
        }

    def wan22_probe_remove(self) -> None:
        for probe in getattr(self, "_wan22_probes", ()):
            probe.remove()
        self._wan22_probes = []

    def wan22_probe_synchronize(self) -> None:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()


def _probe_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value] if value.get("probe") == PROBE_NAME else []
    if isinstance(value, (tuple, list)):
        return [result for item in value for result in _probe_results(item)]
    return []


def _rpc(omni, method: str, **kwargs) -> Any:
    return omni.engine.collective_rpc(method, timeout=120, kwargs=kwargs)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkout_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2], text=True
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--prompt", default="A cat sitting on a table, gently turning its head.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture-calls", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--baseline-runs", type=int, default=2)
    parser.add_argument("--output-type", choices=("pil", "latent"), default="pil")
    parser.add_argument("--output", type=Path, default=Path("wan22_gated_residual_probe.json"))
    args = parser.parse_args(argv)
    for name in ("height", "width", "num_frames", "num_inference_steps", "capture_calls", "baseline_runs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be nonnegative")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine_config = {
        "model": args.model,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "flow_shift": args.flow_shift,
        "worker_extension_cls": EXTENSION_CLASS,
    }
    sampling_config = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "output_type": args.output_type,
    }
    report = {
        "schema_version": 1,
        "probe": PROBE_NAME,
        "environment": {
            "git_revision": _git_revision(),
            "checkout_dirty": _checkout_dirty(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "vllm": _version("vllm"),
            "vllm_omni": _version("vllm-omni"),
            "triton": _version("triton"),
            "diffusers": _version("diffusers"),
            "transformers": _version("transformers"),
        },
        "engine_config": engine_config,
        "sampling_config": sampling_config,
        "prompt": args.prompt,
        "timing_note": "Whole-request diagnostic timings; no fused-kernel comparison. Capture pass is instrumented.",
        "capture_limit_per_block": args.capture_calls,
    }
    start = time.perf_counter()
    omni = Omni(**engine_config)
    report["model_initialization_seconds"] = time.perf_counter() - start

    def generate_once() -> float:
        _rpc(omni, "wan22_probe_synchronize")
        start = time.perf_counter()
        outputs = omni.generate(
            {"prompt": args.prompt, "negative_prompt": ""},
            OmniDiffusionSamplingParams(**sampling_config),
            use_tqdm=False,
        )
        _rpc(omni, "wan22_probe_synchronize")
        duration = time.perf_counter() - start
        if not outputs or not all(output.finished for output in outputs):
            raise RuntimeError("Wan generation did not return completed outputs")
        return duration

    try:
        report["warmup_seconds"] = [generate_once() for _ in range(args.warmup_runs)]
        report["baseline_seconds"] = [generate_once() for _ in range(args.baseline_runs)]
        installed = _probe_results(_rpc(omni, "wan22_probe_install", capture_calls=args.capture_calls))
        if not installed:
            raise RuntimeError("Worker probe installation returned no diagnostic result")
        report["capture_seconds"] = generate_once()
        report["workers"] = _probe_results(_rpc(omni, "wan22_probe_report"))
        if not report["workers"] or not any(worker["records"] for worker in report["workers"]):
            raise RuntimeError("No real gated-residual operands were observed")
        if any(worker["errors"] for worker in report["workers"]):
            raise RuntimeError(f"Incomplete gated-residual capture: {report['workers']}")
    finally:
        try:
            _rpc(omni, "wan22_probe_remove")
        finally:
            omni.close()
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote real Wan gated-residual metadata to {args.output.resolve()}")


if __name__ == "__main__":
    main()
