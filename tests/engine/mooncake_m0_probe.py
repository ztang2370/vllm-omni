# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run the M0 engine experiment with real MRv2/Mooncake and a local model.

Requires a running Mooncake master and MOONCAKE_CONFIG_PATH. The associated
test-only connector uses synthetic HS unless actual model capture is requested.
Run from the repository root with ``python -m tests.engine.mooncake_m0_probe``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

VLLM_REVISION = "4ba787ec9959e566d3b587a796dff07673324da5"
MOONCAKE_REVISION = "6041a609a8c3af35e778f70db344f145c2914980"
MOONCAKE_PACKAGE = "mooncake-transfer-engine-cuda13"
MOONCAKE_VERSION = "0.3.12.post1"


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _build_model(path: Path, seed: int) -> dict[str, str]:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    path.mkdir()
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LlamaForCausalLM(config)
        model.save_pretrained(path, safe_serialization=True)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir()) if p.is_file()}


def _events(path: Path) -> list[dict]:
    events = [json.loads(line) for file in path.glob("*.jsonl") for line in file.read_text().splitlines()]
    return sorted(events, key=lambda event: event["time_ns"])


def _checkpoint_prompt(path: Path) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    template_path = path / "chat_template.json"
    template = json.loads(template_path.read_text())["chat_template"] if template_path.exists() else None
    context = (
        "Mercury is the planet closest to the Sun. A hexagon has six sides. "
        "The Pacific is the largest ocean. Water freezes at zero degrees Celsius. "
    ) * 16
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer the user's question briefly and accurately."},
            {"role": "user", "content": context + "\nHow many sides does a hexagon have?"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        chat_template=template,
        return_dict=False,
    )
    assert 512 < len(prompt) < 1016, f"Expected a 513-1015-token text prompt, got {len(prompt)}"
    return prompt


def _run_request(engine, control_path: Path, case: str, control: dict, prompt: list[int], timeout: float) -> dict:
    from vllm import SamplingParams

    _write_json(control_path, control)
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, detokenize=False, seed=42)
    started = time.monotonic_ns()
    request_id = engine.add_request(f"m0-{case}", {"prompt_token_ids": prompt}, params)
    first_token = None
    final_tokens = None
    steps = 0
    while engine.has_unfinished_requests():
        if (time.monotonic_ns() - started) / 1e9 > timeout:
            engine.abort_request([request_id], internal=True)
            raise TimeoutError(f"Request {case} exceeded {timeout}s")
        outputs = engine.step()
        steps += 1
        for output in outputs:
            tokens = list(output.outputs[0].token_ids)
            if tokens and first_token is None:
                first_token = time.monotonic_ns()
            if output.finished:
                final_tokens = tokens
    assert final_tokens and len(final_tokens) == 8, (case, final_tokens)
    return {
        "case": case,
        "request_id": request_id,
        "control": control,
        "started_ns": started,
        "first_token_ns": first_token,
        "finished_ns": time.monotonic_ns(),
        "steps": steps,
        "token_ids": final_tokens,
    }


def _drain_saves(engine, timeout: float) -> None:
    """Drive the pinned in-process engine until its Store jobs release blocks."""
    # LLMEngine.has_unfinished_requests() only sees client-visible requests.
    # The scheduler also tracks completed-request cleanup and pending saves;
    # Mooncake reports those through worker metadata, not finished_sending.
    scheduler = engine.engine_core.engine_core.scheduler
    deadline = time.monotonic() + timeout
    while scheduler.has_requests():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Mooncake save completion exceeded {timeout}s")
        engine.step()
        time.sleep(0.001)


def _validate(cases: list[dict], events: list[dict], prompt_tokens: int, *, real_hs: bool = False) -> None:
    reference = cases[0]["token_ids"]
    for case in cases:
        assert case["token_ids"] == reference, f"Generated tokens differ: {case['case']}"
        request_events = [event for event in events if event["request_id"] == case["request_id"]]
        scheduled = [event for event in request_events if event["event"] == "scheduled"]
        assert scheduled, f"No scheduling evidence: {case['case']}"
        prefill_tokens = sum(
            min(event["scheduled_tokens"], max(0, prompt_tokens - event["computed_tokens"])) for event in scheduled
        )
        case["prefill_tokens_computed"] = prefill_tokens
        mode = case["control"]["mode"]
        if mode in ("fill", "baseline"):
            assert prefill_tokens == prompt_tokens, (case["case"], prefill_tokens)
            assert not any(event["event"] == "load_started" for event in request_events)
            continue

        loads = [event for event in request_events if event["event"] == "load_started"]
        releases = [event for event in request_events if event["event"] == "joint_released"]
        assert len(loads) == len(releases) == 1, (case["case"], loads, releases)
        release = releases[0]
        assert scheduled[0]["time_ns"] >= release["time_ns"], case["case"]
        assert case["first_token_ns"] >= release["time_ns"], case["case"]
        kv_completions = [event for event in request_events if event["event"] == "kv_completed"]
        assert len(kv_completions) == 1 and not kv_completions[0]["failed"]
        verified = [event for event in request_events if event["event"] == "kv_verified"]
        assert len(verified) == 1 and verified[0]["bytes"] > 0
        assert kv_completions[0]["time_ns"] <= verified[0]["time_ns"] <= release["time_ns"]
        assert verified[0]["blocks"] * 16 == loads[0]["tokens"]
        poisoned = [event for event in request_events if event["event"] == "kv_poisoned"]
        assert len(poisoned) == 1
        assert poisoned[0]["blocks"] == verified[0]["blocks"]
        assert poisoned[0]["bytes"] == verified[0]["bytes"]
        assert poisoned[0]["time_ns"] < verified[0]["time_ns"]
        case["verified_kv_bytes"] = verified[0]["bytes"]
        case["loaded_kv_tokens"] = loads[0]["tokens"]
        case["kv_to_joint_completion_ms"] = (release["time_ns"] - kv_completions[0]["time_ns"]) / 1e6
        if mode == "fail":
            assert release["failed"]
            assert prefill_tokens == prompt_tokens
            assert scheduled[0]["computed_tokens"] == 0
            assert any(event["event"] == "hs_read_failed" for event in request_events)
            assert any(event["event"] == "lookup_bypassed" for event in request_events)
        else:
            expected_load = case["control"].get("max_hit_tokens", 512)
            assert not release["failed"]
            assert loads[0]["tokens"] == expected_load
            assert prefill_tokens == prompt_tokens - expected_load, (case["case"], prefill_tokens)
            hs = [event for event in request_events if event["event"] == "hs_ready"]
            assert len(hs) == 1 and hs[0]["tokens"] == expected_load and hs[0]["bytes"] > 0
            assert hs[0]["time_ns"] <= release["time_ns"]
            assert bool(hs[0].get("real_hs", False)) == real_hs
            case["loaded_hs_bytes"] = hs[0]["bytes"]
            if mode == "delay":
                assert case["kv_to_joint_completion_ms"] >= case["control"]["delay_seconds"] * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for model, events and report")
    parser.add_argument("--store-config", type=Path, default=os.getenv("MOONCAKE_CONFIG_PATH"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--model", type=Path, help="Local pretrained checkpoint; omit to build the tiny seeded model")
    parser.add_argument("--model-id", help="Original model identifier recorded with a local checkpoint")
    parser.add_argument("--revision", help="Exact pretrained source revision")
    parser.add_argument("--capture-hidden-layer", type=int, help="Capture real input embeddings and this hidden layer")
    args = parser.parse_args()
    if args.store_config is None:
        parser.error("--store-config or MOONCAKE_CONFIG_PATH is required")
    if args.model is not None and not args.revision:
        parser.error("--revision is required with --model")
    if args.capture_hidden_layer is not None and args.model is None:
        parser.error("--capture-hidden-layer requires --model")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    event_dir = output / "events"
    event_dir.mkdir()
    control_path = output / "control.json"
    _write_json(control_path, {"mode": "fill"})
    os.environ["MOONCAKE_CONFIG_PATH"] = str(args.store_config.resolve())
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import vllm
    from vllm._version import __commit_id__
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    installed_revision = __commit_id__.removeprefix("g")
    assert len(installed_revision) >= 8 and VLLM_REVISION.startswith(installed_revision), __commit_id__
    assert vllm.__version__ == "0.33.0", vllm.__version__
    mooncake_version = importlib.metadata.version(MOONCAKE_PACKAGE)
    assert mooncake_version == MOONCAKE_VERSION, mooncake_version
    namespace = "m0-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    if args.model is None:
        model_path = output / "tiny-model"
        hashes = _build_model(model_path, seed=42)
        prompt = [3 + (i * 17) % 251 for i in range(513)]
    else:
        model_path = args.model.resolve()
        hashes = {
            name: hashlib.sha256((model_path / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")
            if (model_path / name).exists()
        }
        prompt = _checkpoint_prompt(model_path)
    full_prefix_tokens = (len(prompt) - 1) // 16 * 16
    kv_bytes = (1024 if args.model else 32) * 1024 * 1024
    extra = {
        "m0_control_path": str(control_path),
        "m0_event_dir": str(event_dir),
        "m0_namespace": namespace,
        "cache_prefix": namespace,
        "load_async": True,
        "lookup_async": False,
        "m0_real_hs": args.capture_hidden_layer is not None,
    }
    engine_args = EngineArgs(
        model=str(model_path),
        revision=args.revision,
        language_model_only=args.model is not None,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        seed=42,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        block_size=16,
        kv_cache_memory_bytes=kv_bytes,
        enable_prefix_caching=False,
        enforce_eager=True,
        async_scheduling=False,
        enable_flashinfer_autotune=False,
        distributed_executor_backend="uni",
        moe_backend="triton",
        disable_log_stats=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeM0Connector",
            kv_connector_module_path="tests.engine.mooncake_m0_connector",
            kv_role="kv_both",
            kv_load_failure_policy="recompute",
            kv_connector_extra_config=extra,
        ),
    )
    report = {
        "scope": "MRv2 Thinker engine feasibility; no Talker, waveform, or production backend validation",
        "vllm_version": vllm.__version__,
        "vllm_revision": VLLM_REVISION,
        # The wheel version is observed above; the source revision is the
        # expected release mapping, not an inspected runtime build ID.
        "expected_mooncake_revision": MOONCAKE_REVISION,
        "omni_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "probe_source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("mooncake_m0_probe.py", "mooncake_m0_connector.py")
        },
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", MOONCAKE_PACKAGE)},
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip(),
        "model_seed": 42 if args.model is None else None,
        "model_id": args.model_id or str(model_path),
        # A local path does not verify Hub provenance or hash its weight shards.
        "declared_model_revision": args.revision,
        "capture_hidden_layer": args.capture_hidden_layer,
        "hs_source": "actual_model_captures" if args.capture_hidden_layer is not None else "synthetic_sentinels",
        "model_sha256" if args.model is None else "model_config_sha256": hashes,
        "prompt_token_ids": prompt,
        "store_config": json.loads(args.store_config.read_text()),
        "connector_extra_config": extra,
        "cases": [],
        "passed": False,
    }
    _write_json(output / "report.json", report)
    engine = LLMEngine.from_engine_args(engine_args)
    try:
        assert engine.vllm_config.use_v2_model_runner
        if args.capture_hidden_layer is not None:
            from tests.engine.mooncake_m0_connector import install_qwen_capture

            report["capture_installation"] = engine.apply_model(
                functools.partial(install_qwen_capture, accept_hidden_layer=args.capture_hidden_layer)
            )
        report["engine"] = {
            "runner": "MRv2",
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "block_size": 16,
            "kv_cache_memory_bytes": kv_bytes,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "enforce_eager": True,
            "flashinfer_autotune": False,
            "flashinfer_sampler": os.getenv("VLLM_USE_FLASHINFER_SAMPLER", "1"),
            "moe_backend": "triton",
            "language_model_only": args.model is not None,
        }
        for case, control in (
            ("cold", {"mode": "fill"}),
            ("baseline", {"mode": "baseline"}),
            ("hit", {"mode": "hit", "max_hit_tokens": full_prefix_tokens}),
            ("partial", {"mode": "partial", "max_hit_tokens": 256}),
            ("delay", {"mode": "delay", "delay_seconds": 0.3, "max_hit_tokens": full_prefix_tokens}),
            ("failure", {"mode": "fail", "max_hit_tokens": full_prefix_tokens}),
        ):
            result = _run_request(engine, control_path, case, control, prompt, args.timeout)
            report["cases"].append(result)
            _write_json(output / "report.json", report)
            print(json.dumps(result), flush=True)
            _drain_saves(engine, args.timeout)
        _validate(report["cases"], _events(event_dir), len(prompt), real_hs=args.capture_hidden_layer is not None)
        report["passed"] = True
    finally:
        engine.engine_core.shutdown()
        _write_json(output / "report.json", report)
    print(f"M0 engine probe passed; report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
