# Shared diffusion tensor operations

Public imports come from `vllm_omni.diffusion.layers.ops`. Each operation's
canonical module owns its implementation, dispatch, and custom-op registration.

## Gated residual

```python
from vllm_omni.diffusion.layers.ops import gated_residual

output = gated_residual(residual, branch, gate)
# Reference: residual + branch * gate
```

The [canonical module](modulation/gated_residual.py) carries forward
[PR #6387](https://github.com/vllm-project/vllm-omni/pull/6387) by @Sunbeam23333.
Wan2.2 and HunyuanVideo 1.5 adoption and model qualification are pending;
LTX-2 adoption is a separate follow-up under
[RFC #7382](https://github.com/vllm-project/vllm-omni/issues/7382).

### Contract

The contract follows the five categories of
[RFC #7382 §3](https://github.com/vllm-project/vllm-omni/issues/7382#user-content-integration-rules).

#### 1. Tensor and numerical behavior

- `residual` and `branch` are floating-point tensors with the same non-scalar
  shape. The last dimension is the hidden dimension.
- `gate` is floating-point and broadcasts to that shape without expanding it.
  Scalar gates are valid.
- The output has the residual shape and follows PyTorch dtype promotion.
  Inputs are not mutated, and the output does not alias them.
- For activations shaped `[B, S, C]`, supported gates include:

| Layout | Shape | Meaning |
| --- | --- | --- |
| Global | `[C]` or `[1, 1, C]` | One channel vector for all tokens |
| Per batch item | `[B, 1, C]` | One channel vector per batch item |
| Per token | `[B, S, C]` | One channel vector per token |

- The kernel rounds the product to FP16/BF16 before adding the residual, then
  rounds the output, matching eager PyTorch. The intermediate cast and Triton
  launch option `enable_fp_fusion=False` preserve these two rounding points.

#### 2. Execution behavior

- Supported accelerated variant: the CUDA fast path requires Triton on NVIDIA
  CUDA, matching FP16/BF16 input dtypes and devices, contiguous activations,
  and hidden size 1 through 16384. The gate's last dimension must match the
  hidden size and have stride 1. Higher-rank activations flatten the
  intervening token dimensions into rows. Per-token gates need a constant row
  stride. Gapped `chunk` and `unbind` views are supported when these
  conditions hold, without copying the gate.
- Dispatch: when grad is enabled and any input requires gradients, the public
  helper uses the native expression so autograd records both operations.
  `torch.no_grad()` and `torch.inference_mode()` can still use fusion with
  such inputs.
- Compile/graph modes: the internal custom op and fake implementation support
  `torch.compile` for inference. The custom op has no backward registration;
  use the public helper for the gradient fallback.
- Relocation preserves behavior: this module carries forward
  [PR #6387](https://github.com/vllm-project/vllm-omni/pull/6387) with the
  original authorship, changing only the two review findings.

#### 3. Reference and errors

- Valid inputs outside the fast-path subset use `residual + branch * gate`,
  including mixed dtypes, scalar gates, general broadcasting, and
  noncontiguous activations.
- Invalid activation shapes or gate broadcasting raise `ValueError`;
  non-floating-point tensors raise `TypeError`. Other arithmetic errors follow
  the native expression. CUDA launch errors propagate.

#### 4. Public API

- `gated_residual(residual, branch, gate)` is the public import from
  `vllm_omni.diffusion.layers.ops` (see above).
- The canonical module is the sole custom-op registration owner; no public
  support query is added yet.

#### 5. Adoption

- No model has adopted the shared op yet (model call sites are unchanged).
  Wan2.2 and HunyuanVideo 1.5 adoption and model qualification are pending;
  LTX-2 adoption is a separate follow-up under RFC #7382.
- SANA and MiniMax-H3 VAE retain their own implementations and qualification;
  see Related variants below.

### Related variants

SANA and MiniMax-H3 VAE retain their own implementations and qualification.

| Property | Shared basic operation | SANA variant in #6823 | MiniMax-H3 video VAE |
| --- | --- | --- | --- |
| Arithmetic | Same FP16/BF16 inputs on the fast path | FP16/BF16 gate times update, then residual add | FP32 residual plus FP16 branch times FP32 channel scale |
| Layout | Contiguous activations; global, batch, or token gates | 3D activations; contiguous update and contiguous or transposed-dense residual | Contiguous mixed-dtype tensors, hidden size 2048 |
| Output layout | Contiguous on the fast path | Preserves residual strides, including transposed-dense output | Model-specific output contract |
| Rounding | Product cast and disabled multiply-add contraction | Explicit BF16 rounding and FP16 conversion, with special-value handling | Explicit rounded FP32 multiplication before addition |
| Compilation | Custom op and fake implementation | Uses the native expression in compiled regions | Optional optimization requires eager inference |
| Rejection | Native expression for unsupported inputs | Native fallback; caches launcher failures | Returns `None` so the model can retain its original path |

Sources: [SANA #6823 implementation](https://github.com/vllm-project/vllm-omni/blob/7f70a33537611247258acabef935984434e65929/vllm_omni/diffusion/layers/residual_gate.py),
[H3 scaled residual](../../models/minimax_h3/ops/vae/scaled_residual.py), and
[H3 operator policy](../../models/minimax_h3/ops/README.md).
The shared helper falls back for H3's mixed dtypes and SANA's transposed
activations. Each consumer still needs its own adoption and validation.

## Qualification status (RFC #7382)

| Consumer | Configuration | Status / evidence |
| --- | --- | --- |
| Gated residual, operator level | A100-SXM4-40GB; synthetic tensors, no model weights; FP16/BF16; CPU + CUDA suites; compile coverage; benchmark with correctness check | 87 operator tests pass on A100 (this PR), reproduced on H200: exact forward/backward comparisons, deterministic rounding regressions, compile, strided gates, fused/fallback dispatch |
| Wan2.2 | FP16/BF16; existing `(2,257,4096)` tests with global/per-batch/per-token gates; `(2,17,128)` gapped-gate/compile fixtures; capture the model's actual shapes/dtypes | Pending adoption and model-level qualification |
| HunyuanVideo 1.5 | FP16/BF16; same fixtures; capture the model's actual shapes/dtypes | Pending adoption and model-level qualification |

Remaining gaps: Wan2.2 and HunyuanVideo 1.5 adoption, actual-path assertions
in both models, and the repeated two-model A/B (microbenchmark plus
production traces). The validation runs stay with their current owners.

### Validation

The [operator tests](../../../../tests/diffusion/layers/test_gated_residual.py)
cover exact forward and backward comparisons, rounding, compile, strided gates,
fused/fallback dispatch, and the custom-op schema and fake implementation.
Use the repository's matching vLLM/PyTorch environment and pytest plugins.
CUDA cases require an NVIDIA GPU with FP16/BF16 support and Triton; model
weights are unnecessary. Run from the repository root:

```bash
# Complete operator suite
pytest -q tests/diffusion/layers/test_gated_residual.py

# CUDA CI selection
pytest -q tests/diffusion/layers/test_gated_residual.py \
  -m "core_model and cuda" --run-level=core_model
```

The [microbenchmark](../../../../benchmarks/kernels/gated_residual_benchmarks.py)
checks equality before timing and reports the workload, environment, warmed mean
and standard deviation, and peak extra tensor allocation:

```bash
python -m benchmarks.kernels.gated_residual_benchmarks \
  --batch-size 1 --tokens 32760 --hidden-size 5120 \
  --gate-layout batch --dtype bfloat16 --seed 17 \
  --warmup 25 --iterations 100 --repeats 20
```

Timing includes public-helper validation and dispatch. Peak extra allocation
counts live output and temporary tensors above the resident inputs. Model
outputs and end-to-end performance require separate consumer qualification.
