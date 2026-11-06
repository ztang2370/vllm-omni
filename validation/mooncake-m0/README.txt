Qwen3-Omni Mooncake M0 validation evidence

This branch is review evidence for issue #9908, not a proposed production
backend or a branch to merge into main. The test-only PR is based on commit
dedce6e75dec46f85f677416735047865514b5d5. Its only source change is
 tests/core/sched/test_omni_ar_scheduler_kv_load.py.
The native run used Omni baseline 6e8c1097cd8c73ce799fd1d7f7cd32e656b29b28
plus the exact test/harness files identified by manifest.json.

Results from 2026-11-06 UTC

- 24 CPU cases passed: five scheduler, three capture, sixteen harness cases.
- Native Store registered-buffer and failure checks passed.
- Qwen native cold, baseline, full hit, partial hit, delayed HS, and injected
  HS failure/retry scenarios all passed using the hardened harness.
- KV restored/prompt computed: full hit 512/1, partial hit 256/257,
  delayed HS 512/1, failed HS 512/513, cold and baseline 0/513.
- All six cases returned the same eight greedy text tokens.
- The dedicated Mooncake master was stopped after validation.

The probe verifies restored embeddings and hidden states but does not feed
them into Talker. Execution is sequential, in-process MRv2, TP=1. These results
do not establish duplex behavior, concurrent serving, full audio correctness,
or a performance improvement.

Contents and provenance

- tests/engine/mooncake_m0_*.py and the capture/probe tests preserve the
  executed harness source byte-for-byte, including its generic model mode.
- qwen/report.json and qwen/events/ contain fresh Qwen results and event traces.
- cpu-tests.log, store-report.json and master-version.log contain test output.
- packages.txt lists relevant installed versions and tested hardware.
- manifest.json identifies original source hashes and published artifact hashes.

Published artifacts replace local checkout, environment, and model-root paths
with <checkout>, <venv>, <python-runtime>, and <repo>. Tensor/checksum values,
request identifiers, event timestamps, token counts, and source hashes are
unchanged. Source files themselves are not redacted. Model weights, package
caches, unrelated job logs, and other-model results are not included.

Reproduction

Use this branch's original tests/engine paths so tests.engine imports resolve.
Prepare a CUDA development environment with Python 3.12, the versions listed
in packages.txt, Omni installed, and pytest-mock. The vLLM wheel is pinned to:
https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
Use matching Python headers, a CUDA compiler and Ninja. The recorded run used
nvcc 13.1.115, Python 3.12.15, and an A100 80 GB GPU. If the master cannot find
libcudart.so.13, include the environment's nvidia/cu13/lib on LD_LIBRARY_PATH.

Run CPU tests from the repository root:

python -m pytest tests/core/sched/test_omni_ar_scheduler_kv_load.py tests/engine/test_mooncake_m0_capture.py tests/engine/test_mooncake_m0_probe.py -q -m 'core_model and cpu' --run-level core_model

Start a dedicated master in another terminal:

mooncake_master --rpc_port=50051 --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 --http_metadata_server_port=8080 --metrics_port=9003

The checked-in Store config uses those ports. The recorded run used isolated
ports 51051/18080/19003 with the same single-host TCP topology and buffer sizes.

python tests/engine/mooncake_m0_store_probe.py --output /tmp/mooncake-store-report.json

Download Qwen/Qwen3-Omni-30B-A3B-Instruct at revision
26291f793822fb6be9555850f06dfe95f2d7e695 into /path/to/qwen3-omni, then run:

VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=/tmp/mooncake-cache python -m tests.engine.mooncake_m0_probe --store-config tests/engine/mooncake_m0_store.json --output-dir /tmp/mooncake-qwen-result --model /path/to/qwen3-omni --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct --revision 26291f793822fb6be9555850f06dfe95f2d7e695 --capture-hidden-layer 24 --timeout 300

Use a new output directory for each run. Stop the dedicated master afterward.
The harness records the supplied model revision and hashes the config/index;
it does not independently hash every pretrained weight shard.

AI assistance: Codex reviewed the tests and harness, ran validation, prepared
this evidence, and drafted the PR description. The contributor reviewed the
proposed tests and description before authorizing the push.
