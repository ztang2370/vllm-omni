# Wan2.2 gated-residual contract validation

This is model-side preparation for Phase 2/R1 in [#7382](https://github.com/vllm-project/vllm-omni/issues/7382), building on [#6387](https://github.com/vllm-project/vllm-omni/pull/6387). It records the operands used by the existing Wan forward and protects its arithmetic contract. The production model and shared kernel are unchanged.

## Tests and probe

Run from the repository root in a working vLLM-Omni development environment:

```bash
pytest -q tests/diffusion/models/wan2_2/test_wan22_transformer_block.py \
    tests/benchmarks/test_wan22_gated_residual_probe.py -m "core_model and cpu"
```

The six production-forward tests substitute deterministic child modules while retaining `WanTransformerBlock.forward`. They cover batch/token gate indexing, separate attention/FFN gates, the ungated cross-attention update, FP32 gate promotion with BF16 output restoration, and BF16 product rounding before residual addition. Ten diagnostic tests cover observation fidelity, capture bounds, cleanup and CLI handling. All 16 tests passed; they establish CPU contracts, not CUDA kernel parity.

The probe uses the existing worker-extension RPC interface to install bounded hooks on the first Wan block of each loaded transformer. `TorchDispatchMode` follows the actual branch multiply and the add consuming that product. It records tensor metadata through the final cast, without reconstructing gates or retaining tensor contents. Hooks are removed before shutdown. Eager execution is required.

For a fresh CUDA environment, install matching vLLM first, then this checkout:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.29.0 --torch-backend=auto
uv pip install -e . pytest pytest-xdist pytest-asyncio
```

Check installed versions against the recorded environment below; automatic dependency resolution can select different builds. Pin the model separately:

```bash
CHECKPOINT="$PWD/checkpoints/Wan2.2-TI2V-5B-Diffusers"
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --revision b8fff7315c768468a5333511427288870b2e9635 \
    --local-dir "$CHECKPOINT"

python -m benchmarks.kernels.wan22_gated_residual_probe \
    --model "$CHECKPOINT" \
    --prompt "A cat sitting on a table, gently turning its head." \
    --height 512 --width 512 --num-frames 9 --num-inference-steps 2 \
    --guidance-scale 5 --seed 42 --flow-shift 12 --output-type pil \
    --capture-calls 2 --warmup-runs 1 --baseline-runs 2 \
    --output wan22_gated_residual_probe.json
```

The JSON includes the production commit, checkout dirty state, probe source SHA-256, UTC timestamp and package versions. It separates uninstrumented warmup/baseline requests from the instrumented capture request. Its timings describe complete requests on that checkout, including orchestration and output handling. They do not measure a fused kernel or establish a speedup.

## Observed H100 capture

| Setting | Recorded value |
| --- | --- |
| Production source base | `7e520f9cc47e6884520ea8f3f9ab94ab97b34780` |
| GPU / driver | NVIDIA H100 80GB HBM3 / `595.84` |
| Python | `3.12.3` |
| PyTorch / CUDA build / Triton | `2.13.0+cu132` / `13.2` / `3.7.1` |
| vLLM / Diffusers / Transformers | `0.29.0` / `0.40.0` / `5.14.1` |
| vLLM-Omni package metadata | `0.29.0.dev0+g7e520f9c` |
| Model | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| Model revision | `b8fff7315c768468a5333511427288870b2e9635` |
| Runtime | BF16, eager, native `FLASH_ATTN`, one GPU; no offload, sequence parallelism or CFG parallelism |

The tested fork lacked release tags, so its editable installation used `VLLM_OMNI_VERSION_OVERRIDE=0.29.0.dev0+g7e520f9c`. The source commit above identifies the actual production code.

The workload is the command above. `transformer.blocks.0` executed four times: positive and negative CFG passes across two denoising steps. Capturing its first two calls produced four records, one self-attention and one FFN residual update per call.

| Operand | Observed metadata at both sites |
| --- | --- |
| Residual and branch | BF16, `[1, 768, 3072]`, contiguous, same CUDA device |
| Gate | BF16, `[1, 1, 3072]`, stride `[18432, 3072, 1]` |
| Gate storage offset | Self-attention: `6144`; FFN: `15360` |
| Product, sum and final output | BF16 |

The gates are offset views but report contiguous: dimensions of size one do not impose a contiguity constraint. These records provide no evidence of an FP32-gate or noncontiguous-gate fallback in this workload.

The TI2V checkpoint name does not imply token-wise modulation for every request. In [the Wan pipeline](../../vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py), token-wise timestep expansion requires both `expand_timesteps` and a `latent_condition`. This text-only run used the standard batch-gate path. The CPU tests cover token-wise indexing; a real conditioned run is still required to qualify that path.

The ordinary video example also completed a decoded generation and MP4 export at 512×512, 33 frames and 50 steps. The saved H.264 video decoded to 33 frames at 16 FPS. Reproduce that separate smoke run with:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model "$CHECKPOINT" \
    --prompt "A cat sitting on a table, gently turning its head." \
    --negative-prompt "" --height 512 --width 512 --num-frames 33 \
    --num-inference-steps 50 --guidance-scale 5 --flow-shift 12 \
    --seed 42 --fps 16 --enforce-eager --vae-use-tiling \
    --output wan22-cat.mp4
```

## Relation to the shared kernel

Inspection of #6387 at `7da8d1447ac0ce8d8a545e1427d376e580ab7fd8` indicates that the captured tensors satisfy its fast-path checks: matching CUDA devices and BF16 dtypes, contiguous residual/branch, gate last stride one, and hidden size `3072 <= 16384`. With all leading gate dimensions equal to one, its layout classifier selects the global-gate case. Batch size one does not qualify multi-batch gate indexing.

This is eligibility by inspection. The probe ran the existing eager expression, so it does not demonstrate dispatch into #6387, numerical parity with that kernel, latency improvement or VRAM savings. GPU kernel correctness, actual dispatch, representative workloads and end-to-end A/B measurements remain necessary before making those claims. HunyuanVideo1.5 qualification is subsequent R1 work; LTX-2 belongs to the separate R2 follow-up.
