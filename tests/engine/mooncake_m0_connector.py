# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test-only joint completion adapter for the pinned Mooncake M0 experiment.

KV uses the real vLLM Mooncake Store connector. The tiny-model probe uses
synthetic HS arrays; the Qwen probe captures real embeddings and the selected
Thinker layer. This adapter probes engine waiting and recovery, not Talker.
It deliberately depends on vLLM 0.33's private ``load_specs`` for prefix clamping.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole, KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
from vllm.v1.kv_cache_interface import group_kernel_blocks

_QWEN_CAPTURES: dict[str, torch.Tensor] = {}


def install_qwen_capture(model, *, accept_hidden_layer: int):
    """Install eager-only hooks matching Omni's pre-layer capture convention."""
    language_model = model.get_language_model()
    layers = language_model.model.layers
    assert 0 < accept_hidden_layer < len(layers)

    def capture(name):
        def hook(_module, args):
            _, hidden, residual = args
            # Omni captures before layer[index], adding the residual deferred
            # by vLLM until its next RMSNorm. Index 0 is the input embedding.
            _QWEN_CAPTURES[name] = hidden.detach().clone() if residual is None else (hidden + residual).detach()

        return hook

    layers[0].register_forward_pre_hook(capture("embedding"))
    layers[accept_hidden_layer].register_forward_pre_hook(capture("hidden"))
    return {
        "capture_indices": [0, accept_hidden_layer],
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "language_model_class": f"{type(language_model).__module__}.{type(language_model).__name__}",
    }


class MooncakeM0Connector(MooncakeStoreConnector):
    """Single-worker, sequential-request M0 probe; not a supported backend."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        assert vllm_config.parallel_config.tensor_parallel_size == 1
        assert vllm_config.use_v2_model_runner
        assert not vllm_config.cache_config.enable_prefix_caching
        assert len(kv_cache_config.kv_cache_groups) == 1
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._real_hs = extra.get("m0_real_hs", False)
        self._hs_references: dict[str, dict[str, torch.Tensor]] = {}
        self._restored_hs: dict[str, dict[str, torch.Tensor]] = {}
        self._control_path = Path(extra["m0_control_path"])
        self._event_dir = Path(extra["m0_event_dir"])
        self._event_dir.mkdir(parents=True, exist_ok=True)
        self._namespace = extra["m0_namespace"]
        self._event_role = "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
        self._controls: dict[str, dict] = {}
        self._load_states: dict[str, str] = {}
        self._failed_attempts: set[str] = set()
        self._loads: dict[str, dict] = {}
        self._completed_kv: dict[str, tuple[bool, float]] = {}
        self._saved_hs: set[str] = set()
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_worker is not None:
            assert self.connector_worker.hash_block_size == self.connector_worker.block_size
        else:
            assert self.connector_scheduler._hash_block_size == self.connector_scheduler._block_size
        self._event("initialized", runner_v2=vllm_config.use_v2_model_runner, real_hs=self._real_hs)

    def _event(self, event, request_id=None, **fields):
        record = {"event": event, "request_id": request_id, "time_ns": time.monotonic_ns(), **fields}
        path = self._event_dir / f"{self._event_role}-{os.getpid()}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        request_id = request.request_id
        if request_id not in self._controls:
            self._controls[request_id] = json.loads(self._control_path.read_text())
        control = self._controls[request_id]
        if request_id in self._failed_attempts or control["mode"] == "baseline":
            self.connector_scheduler.load_specs.pop(request_id, None)
            self._event("lookup_bypassed", request_id, computed_tokens=num_computed_tokens, mode=control["mode"])
            return 0, False
        count, async_load = super().get_num_new_matched_tokens(request, num_computed_tokens)
        original_count = count
        if count:
            block_size = self.connector_scheduler._block_size
            cap = control.get("max_hit_tokens", count + num_computed_tokens)
            selected_end = min(count + num_computed_tokens, cap)
            selected_end -= selected_end % block_size
            count = max(0, selected_end - num_computed_tokens)
            spec = self.connector_scheduler.load_specs[request_id]
            spec.kvpool_cached_tokens = num_computed_tokens + count
            # M0 only uses ordinary full-attention, block-aligned cache entries.
            assert all(b.num_tokens == num_computed_tokens + original_count for b in spec.tail_key_boundaries)
            spec.tail_key_boundaries = tuple(
                replace(boundary, num_tokens=spec.kvpool_cached_tokens) for boundary in spec.tail_key_boundaries
            )
            self._load_states[request_id] = "pending" if count else "miss"
        self._event(
            "lookup",
            request_id,
            mode=control["mode"],
            kv_tokens=original_count,
            selected_tokens=count,
            computed_tokens=num_computed_tokens,
        )
        return count, bool(count) and async_load

    def build_connector_meta(self, scheduler_output):
        computed = {r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        for request_id, count in scheduler_output.num_scheduled_tokens.items():
            state = self._load_states.get(request_id)
            assert state != "pending", f"Model scheduled before joint completion: {request_id}"
            self._event(
                "scheduled",
                request_id,
                computed_tokens=computed[request_id],
                scheduled_tokens=count,
                load_state=state,
            )
        metadata = super().build_connector_meta(scheduler_output)
        metadata.m0_controls = dict(self._controls)
        for request_id in scheduler_output.finished_req_ids:
            self._controls.pop(request_id, None)
            self._load_states.pop(request_id, None)
            self._failed_attempts.discard(request_id)
        return metadata

    def update_connector_output(self, connector_output):
        super().update_connector_output(connector_output)
        for request_id in connector_output.failed_recving or ():
            self._failed_attempts.add(request_id)
            self._load_states[request_id] = "failed"
            self._event("failure_received", request_id)
        for request_id in connector_output.finished_recving or ():
            if request_id not in self._failed_attempts:
                self._load_states[request_id] = "ready"
            self._event("completion_received", request_id, load_state=self._load_states[request_id])

    def _hs_key(self, block_hash):
        digest = hashlib.sha256(str(block_hash).encode()).hexdigest()
        kind = "qwen-hs" if self._real_hs else "synthetic-hs"
        return f"{self._namespace}:{kind}:{digest}"

    def register_kv_caches(self, kv_caches):
        assert kv_caches
        assert set(kv_caches) == set(self._kv_cache_config.kv_cache_groups[0].layer_names)
        super().register_kv_caches(kv_caches)
        self._kv_caches = {
            name: group_kernel_blocks(cache, self.connector_worker.num_blocks)
            for name, cache in sorted(kv_caches.items())
        }

    def _kv_digest(self, block_id):
        digest = hashlib.sha256()
        size = 0
        for name, cache in self._kv_caches.items():
            data = cache[block_id].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            digest.update(name.encode())
            digest.update(data)
            size += len(data)
        return digest.hexdigest().encode(), size

    @staticmethod
    def _hs_bytes(key, rows):
        # Two arrays of shape [rows, 4], float32. Their content depends on the
        # same prefix hash as KV and is exactly checkable after a Store read.
        seed = hashlib.sha256(key.encode()).digest()[0]
        values = [float(seed + i) for i in range(rows * 8)]
        return b"M0HS" + struct.pack("<II", rows, 4) + struct.pack(f"<{len(values)}f", *values)

    def _save_hs_bytes(self, key, start, end):
        if not self._real_hs:
            return self._hs_bytes(key, end - start)
        assert set(_QWEN_CAPTURES) == {"embedding", "hidden"}, "Real Qwen capture hooks did not run"
        assert all(tensor.shape[0] >= end for tensor in _QWEN_CAPTURES.values()), "M0 requires unchunked prefill"
        tensors = {name: tensor[start:end].detach().cpu().contiguous() for name, tensor in _QWEN_CAPTURES.items()}
        self._hs_references[key] = tensors
        return save_tensors(tensors)

    def _read_hs_bytes(self, key, payload, rows):
        if not self._real_hs:
            return payload == self._hs_bytes(key, rows), None
        if not payload:
            return False, None
        try:
            tensors = load_tensors(payload)
        except SafetensorError:
            return False, None
        reference = self._hs_references[key]
        valid = set(tensors) == set(reference) and all(
            expected.shape[0] == rows
            and tensors[name].shape == expected.shape
            and tensors[name].dtype == expected.dtype
            and torch.equal(tensors[name], expected)
            for name, expected in reference.items()
        )
        return valid, tensors if valid else None

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        self._controls.update(metadata.m0_controls)
        for request in metadata.requests:
            spec = request.load_spec
            if spec is None or not spec.can_load:
                continue
            block_size = self.connector_worker.block_size
            start = spec.vllm_cached_tokens // block_size
            end = spec.kvpool_cached_tokens // block_size
            keys = [self._hs_key(h) for h in request.block_hashes[start:end]]
            assert len(keys) == end - start
            self._loads[request.req_id] = {
                "keys": keys,
                "block_ids": request.block_ids[0][start:end],
                "rows": block_size,
                "tokens": (end - start) * block_size,
            }
            # Reused GPU slots may still contain the same prompt's old KV.
            # Destroy it before submitting the native read so digest equality
            # afterwards proves transfer, rather than incidental residency.
            poisoned_bytes = 0
            for key, block_id in zip(keys, request.block_ids[0][start:end], strict=True):
                for cache in self._kv_caches.values():
                    cache[block_id].zero_()
                # The device-to-host digest copy fences the poison writes
                # before the background Mooncake transfer can touch the slots.
                digest, size = self._kv_digest(block_id)
                expected = self.connector_worker.store.get(key + ":kv-sha256")
                assert expected and digest != expected, "KV destination was not poisoned"
                poisoned_bytes += size
            self._event("kv_poisoned", request.req_id, blocks=len(keys), bytes=poisoned_bytes)
            self._event("load_started", request.req_id, tokens=(end - start) * block_size)
        return super().start_load_kv(forward_context, **kwargs)

    def wait_for_save(self):
        super().wait_for_save()
        metadata = self._get_connector_metadata()
        for request in metadata.requests:
            if not request.can_save:
                continue
            # This diagnostic copy is intentionally synchronous. Fence the
            # same forward completion event used by the native KV save thread.
            if request.current_event is not None:
                request.current_event.synchronize()
            block_size = self.connector_worker.block_size
            end = min(request.token_len_chunk, request.num_prompt_tokens) // block_size
            for index, block_hash in enumerate(request.block_hashes[:end]):
                key = self._hs_key(block_hash)
                if key in self._saved_hs:
                    continue
                payload = self._save_hs_bytes(key, index * block_size, (index + 1) * block_size)
                result = self.connector_worker.store.put(key, payload)
                assert result == 0, f"HS write failed: {result}"
                kv_digest, kv_bytes = self._kv_digest(request.block_ids[0][index])
                assert self.connector_worker.store.put(key + ":kv-sha256", kv_digest) == 0
                self._saved_hs.add(key)
                self._event(
                    "hs_saved",
                    request.req_id,
                    key=key,
                    bytes=len(payload),
                    rows=block_size,
                    kv_bytes=kv_bytes,
                    real_hs=self._real_hs,
                )

    def get_transfer_results(self, finished_req_ids):
        for request_id in finished_req_ids:
            self._restored_hs.pop(request_id, None)
        result = super().get_transfer_results(finished_req_ids)
        now = time.monotonic()
        for request_id in result.finished_recving:
            self._completed_kv[request_id] = (request_id in result.failed_recving, now)
            self._event("kv_completed", request_id, failed=request_id in result.failed_recving)
        completed, failed = set(), set()
        for request_id, (kv_failed, completed_at) in list(self._completed_kv.items()):
            control = self._controls[request_id]
            delay = control.get("delay_seconds", 0.0) if control["mode"] == "delay" else 0.0
            if now - completed_at < delay:
                continue
            load = self._loads.pop(request_id)
            hs_failed = kv_failed
            loaded_bytes = 0
            restored = []
            if not kv_failed:
                kv_bytes = 0
                for key, block_id in zip(load["keys"], load["block_ids"], strict=True):
                    digest, size = self._kv_digest(block_id)
                    assert digest == self.connector_worker.store.get(key + ":kv-sha256"), "Restored KV differs"
                    kv_bytes += size
                self._event("kv_verified", request_id, blocks=len(load["keys"]), bytes=kv_bytes)
                for key in load["keys"]:
                    read_key = key + ":deliberately-missing" if control["mode"] == "fail" else key
                    payload = self.connector_worker.store.get(read_key)
                    valid, tensors = self._read_hs_bytes(key, payload, load["rows"])
                    if not valid:
                        hs_failed = True
                        self._event("hs_read_failed", request_id, key=read_key)
                        break
                    loaded_bytes += len(payload)
                    if tensors is not None:
                        restored.append(tensors)
            if hs_failed:
                failed.add(request_id)
            else:
                if self._real_hs:
                    self._restored_hs[request_id] = {
                        name: torch.cat([block[name] for block in restored]) for name in ("embedding", "hidden")
                    }
                    assert all(t.shape[0] == load["tokens"] for t in self._restored_hs[request_id].values())
                self._event("hs_ready", request_id, tokens=load["tokens"], bytes=loaded_bytes, real_hs=self._real_hs)
            completed.add(request_id)
            del self._completed_kv[request_id]
            self._event("joint_released", request_id, failed=hs_failed)
        return KVConnectorTransferResults(
            finished_sending=result.finished_sending,
            finished_recving=completed,
            failed_recving=failed,
        )
