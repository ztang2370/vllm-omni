# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The M0 report must reject incomplete or prematurely released cache loads."""

import pytest

from tests.engine import mooncake_m0_probe as probe
from tests.engine.mooncake_m0_probe import _validate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _trace(mode="hit"):
    """A cold baseline followed by one externally loaded request."""
    loaded = 256 if mode == "partial" else 512
    failed = mode == "fail"
    release_time = 400_000_000 if mode == "delay" else 90
    control = {"mode": mode, "max_hit_tokens": loaded}
    if mode == "delay":
        control["delay_seconds"] = 0.3
    cases = [
        {"case": "baseline", "request_id": "cold", "control": {"mode": "baseline"}, "token_ids": [1, 2]},
        {
            "case": mode,
            "request_id": "loaded",
            "control": control,
            "token_ids": [1, 2],
            "first_token_ns": release_time + 20,
        },
    ]
    events = [
        {
            "request_id": "cold",
            "event": "scheduled",
            "time_ns": 1,
            "computed_tokens": 0,
            "scheduled_tokens": 513,
        }
    ]

    def add(event, time_ns, **fields):
        events.append({"request_id": "loaded", "event": event, "time_ns": time_ns, **fields})

    add("kv_poisoned", 20, blocks=loaded // 16, bytes=1024)
    add("load_started", 30, tokens=loaded)
    add("kv_completed", 40, failed=False)
    add("kv_verified", 50, blocks=loaded // 16, bytes=1024)
    if failed:
        add("hs_read_failed", 60)
        add("lookup_bypassed", release_time + 1)
    else:
        add("hs_ready", release_time - 1, tokens=loaded, bytes=256, real_hs=True)
    add("joint_released", release_time, failed=failed)
    add(
        "scheduled",
        release_time + 10,
        computed_tokens=0 if failed else loaded,
        scheduled_tokens=513 if failed else 513 - loaded,
    )
    return cases, events


@pytest.mark.parametrize("mode", ["hit", "partial", "delay", "fail"])
def test_validation_accepts_joint_completion_and_recomputation(mode):
    cases, events = _trace(mode)
    _validate(cases, events, 513, real_hs=True)
    assert cases[1]["prefill_tokens_computed"] == (513 if mode == "fail" else 513 - cases[1]["loaded_kv_tokens"])


@pytest.mark.parametrize("early_output", [False, True], ids=["model_scheduled", "client_output"])
def test_validation_rejects_work_before_joint_release(early_output):
    cases, events = _trace("delay")
    if early_output:
        cases[1]["first_token_ns"] = 100
    else:
        next(event for event in events if event["request_id"] == "loaded" and event["event"] == "scheduled")[
            "time_ns"
        ] = 100
    with pytest.raises(AssertionError):
        _validate(cases, events, 513, real_hs=True)


@pytest.mark.parametrize("missing_event", ["kv_poisoned", "kv_verified", "hs_ready"])
def test_validation_requires_transfer_evidence(missing_event):
    cases, events = _trace()
    events = [event for event in events if event["event"] != missing_event]
    with pytest.raises(AssertionError):
        _validate(cases, events, 513, real_hs=True)


@pytest.mark.parametrize("late_event", ["kv_verified", "hs_ready"])
def test_validation_rejects_release_before_both_tensors_are_ready(late_event):
    cases, events = _trace()
    next(event for event in events if event["event"] == late_event)["time_ns"] = 91
    with pytest.raises(AssertionError):
        _validate(cases, events, 513, real_hs=True)


def test_validation_rejects_failed_load_that_skips_prefill():
    cases, events = _trace("fail")
    scheduled = next(event for event in events if event["request_id"] == "loaded" and event["event"] == "scheduled")
    scheduled.update(computed_tokens=512, scheduled_tokens=1)
    with pytest.raises(AssertionError):
        _validate(cases, events, 513, real_hs=True)


def test_validation_rejects_changed_generated_tokens():
    cases, events = _trace()
    cases[1]["token_ids"] = [1, 3]
    with pytest.raises(AssertionError, match="Generated tokens differ"):
        _validate(cases, events, 513, real_hs=True)


class _SavingEngine:
    def __init__(self, pending_saves: int):
        self.pending_saves = pending_saves
        self.engine_core = self
        self.scheduler = self

    def has_requests(self) -> bool:
        return self.pending_saves > 0

    def step(self) -> None:
        self.pending_saves -= 1


@pytest.mark.parametrize("pending_saves", [0, 3])
def test_drain_saves_drives_engine_until_scheduler_is_idle(pending_saves):
    engine = _SavingEngine(pending_saves)
    probe._drain_saves(engine, timeout=1)
    assert engine.pending_saves == 0


def test_drain_saves_times_out_if_store_never_completes(monkeypatch):
    times = iter([0, 2])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(times))
    engine = _SavingEngine(1)
    with pytest.raises(TimeoutError, match="save completion exceeded"):
        probe._drain_saves(engine, timeout=1)
    assert engine.pending_saves == 1
