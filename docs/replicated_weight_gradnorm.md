# Replicated weights are counted `tp_size` times in the global gradient norm

## Summary

`transformer_engine.pytorch.Linear` marks **every** weight as tensor-model-parallel,
including weights that are explicitly replicated (`parallel_mode=None`, which is what
Megatron's `"duplicated"` maps to). Megatron reads that attribute to decide whether a
parameter enters the global gradient-norm sum once, or once per TP rank. Replicated
weights are therefore counted `tp_size` times and the reported gradient norm is
inflated.

Measured on two MLA models at TP=8, each with a constant per-iteration ratio:

| model | stock | fixed | inflation |
|---|---|---|---|
| DeepSeek-V4-Flash (4 layers) | 3.763 | 2.472 | **1.52×** |
| DeepSeek-V3 (6 layers, Megatron's own MLA) | 12.228 | 9.011 | **1.36×** |

The DeepSeek-V3 run uses Megatron's `pretrain_gpt.py` with
`--multi-latent-attention --transformer-impl transformer_engine` and no third-party
model code, so the behaviour is reproducible with upstream components alone.

**This is primarily a diagnostic bug.** The gradient norm is the number practitioners
use to judge training health, tune `--clip-grad`, detect instability, and compare
against reference curves — and it is wrong by 52% here. Its effect on the *weights*
is much smaller, and depends entirely on the optimizer (see
[Impact on training](#impact-on-training)).

## Mechanism

`transformer_engine/pytorch/module/linear.py`, `Linear.reset_parameters`:

```python
for weight in self.weight_names:
    set_tensor_model_parallel_attributes(
        tensor=getattr(self, weight),
        is_parallel=True,                                  # unconditional
        dim=1 if self.parallel_mode == "row" else 0,
        stride=1,
    )
```

`parallel_mode` selects the *dim* but never whether to mark at all, so a non-parallel
linear is still labelled `tensor_model_parallel=True`.

Megatron consumes it in `param_is_not_tensor_parallel_duplicate()`:

```python
if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
    return True              # -> admitted on EVERY TP rank
return tp_group.rank() == 0  # -> replicated: admitted once
```

which gates `get_grads_for_norm()`. The norm is then assembled as a sum of squares
across ranks, so a replicated weight's contribution is added `tp_size` times:

```python
total_norm = local_l2_norm(grads_for_norm) ** 2
all_reduce(total_norm, op=SUM, group=grad_stats_parallel_group)
total_norm = total_norm ** 0.5
```

Two details confirm the intent:

* TE never *reads* the attribute back — it is set purely for downstream consumers.
* TE's own `_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS` declares
  `{'tensor_model_parallel': False, 'partition_dim': -1, 'partition_stride': 1}`;
  the code overrides its own stated default for the non-parallel case.

Megatron does not correct it afterwards: the post-init loop in
`megatron/core/extensions/transformer_engine.py` sets only `allreduce` and
`sequence_parallel`.

Note there is no separate "duplicated" state. The attribute set is entirely TP-scoped
and TE's `Linear` has no data-parallel concept at all (`sequence_parallel`, `tp_group`,
`tp_size`, `parallel_mode`). Replicated *is* `tensor_model_parallel=False`.

## How large is it

$$\text{inflation} = \sqrt{1 + (\text{TP} - 1)\,f}$$

where `f` is the duplicated weights' share of the **true squared norm** — gradient
*energy*, not parameter count. The distinction matters: in DeepSeek-V4-Flash the
replicated projections are ~1.5% of parameters but hold ~19% of gradient energy,
because they see the full hidden state and every head's gradient flows back through
them.

| f | TP=2 | TP=4 | TP=8 | TP=16 | TP=32 |
|---|---|---|---|---|---|
| 1.5% (typical replicated weight) | 1.01 | 1.02 | 1.05 | 1.09 | 1.12 |
| 19% (MLA down-projections) | 1.09 | 1.25 | **1.53** | 1.96 | 2.62 |

Inverting the measured ratios gives `f` directly, and both models land far above their
parameter share (~0.3%), confirming the effect tracks gradient energy rather than
parameter count:

| model | measured ratio @ TP=8 | implied f |
|---|---|---|
| DeepSeek-V4-Flash | 1.522 | 18.8% |
| DeepSeek-V3 | 1.361 | 12.2% |

So the same bug is invisible in most models and significant in MLA at large TP.

## Which axis is affected

Only tensor parallelism.

| axis | protection |
|---|---|
| DP | Sharding. With `--use-distributed-optimizer` each rank's `get_parameters()` returns only its shard, so a gradient enters the sum on exactly one DP rank. Without it, the norm group is the model-parallel group, excluding DP. |
| TP | `param_is_not_tensor_parallel_duplicate()` — the attribute above, and the **only** guard on this axis. |

## Impact on training

Megatron's `--clip-grad` is *norm*-based and *global*: one scalar for the whole model,
applied uniformly.

```python
clip_coeff = max_norm / (total_norm + 1.0e-6)
if clip_coeff < 1.0:
    multi_tensor_scale(grads, grads, clip_coeff)
```

This is a **rescale, not a clip** — direction and the relative weighting between
parameters are preserved exactly. Three properties together bound the damage:

1. **global**, not per-parameter — a wrong scalar is a magnitude error, not a
   relative-weighting error;
2. **norm-based**, not value-based — no element-wise truncation, so direction is
   untouched;
3. paired with a **scale-invariant optimizer**, the magnitude error is absorbed.

It also only engages at all when the inflated norm crosses the threshold:

| condition | effect |
|---|---|
| `--clip-grad 0` | none — the norm is not even computed |
| both true and inflated norm below threshold | none — `clip_coeff >= 1`, gated off |
| inflated norm crosses the threshold | gradients scaled by `1/inflation` |

DeepSeek-V4-Flash is in the third regime: true norm ~2.47 against `--clip-grad 1.0`,
so clipping fires every step, with coefficient `0.266` instead of `0.405` — gradients
34% smaller than intended.

What that does to the weights depends on the optimizer:

| optimizer | scale invariance | impact |
|---|---|---|
| **SGD** | none — update is linear in the gradient | **full**: 34% smaller steps. Also, weight decay is coupled and applied *after* clipping, so the decay term is not scaled and becomes 1.52× stronger relative to the gradient. |
| **Adam / AdamW** | approximate | **second-order**. `m -> c·m`, `v -> c²·v`, so `update = lr·m̂/(√v̂ + ε/c)` — the `c` cancels except through `ε`. Residue: `c` varies per step so cancellation across accumulated moments is imperfect; plus the first steps before moments equilibrate. |
| **Muon** | exact | **none** for 2D params. Newton-Schulz maps `B = UΣVᵀ -> UVᵀ`, discarding the singular values, and the update magnitude comes from a shape-dependent `get_muon_scale_factor(size[0], size[1])`. Scaling `B` by `c` leaves `UVᵀ` unchanged. |

Megatron's default is Adam, so for most users the weight-level effect is small — but
the reported norm is wrong regardless, and an SGD configuration sees the full effect.

## Which models are exposed

For a standard transformer every linear is column- or row-parallel, so `is_parallel=True`
is correct and the bug cannot trigger.

It requires an architecture that *deliberately replicates* weights carrying real
gradient energy. Multi-head Latent Attention is exactly that — the low-rank
down-projections are replicated because the KV latent is shared by every head (each TP
rank needs the whole latent), and because the down-projection feeds a column-parallel
up-projection that requires the complete input on every rank.

Megatron's own MLA does this explicitly
(`megatron/core/transformer/multi_latent_attention.py`):

```python
if submodules.linear_q_down_proj in [TELinear]:
    q_down_proj_kwargs['parallel_mode'] = 'duplicated'
...
if submodules.linear_kv_down_proj in [TELinear]:
    kv_down_proj_kwargs['parallel_mode'] = 'duplicated'
```

**Scope:** any Megatron MLA model takes this path. DeepSeek-V3 has been measured
directly (see Evidence) using Megatron's own `pretrain_gpt.py`, confirming the
behaviour is not specific to any downstream model implementation. DeepSeek-V2 and V3.2
share the same spec path and are expected to behave identically, though they have not
been run.

## Evidence

DeepSeek-V4-Flash, 4 layers, TP=8, EP=8, PP=1, DP=1, bf16, fixed seed (deterministic,
identical across reruns). Two linear backends, to show the behaviour is not specific
to one implementation.

| backend | | it 1 | it 2 | it 3 | it 4 | it 5 |
|---|---|---|---|---|---|---|
| Lumen | current | 3.763 | 3.837 | 3.720 | 3.593 | 3.748 |
| Lumen | fixed | 2.472 | 2.520 | 2.445 | 2.359 | 2.462 |
| Lumen | ratio | 1.522 | 1.523 | 1.521 | 1.523 | 1.522 |
| TE | current | 4.098 | 4.177 | 4.079 | 3.921 | 4.112 |
| TE | fixed | 2.634 | 2.688 | 2.623 | 2.520 | 2.644 |
| TE | ratio | 1.556 | 1.554 | 1.555 | 1.556 | 1.555 |

The constant ratio across iterations is the signature of a counting error rather than a
numerical one. The two backends differ slightly only because the runs have different
random initialisation.

Loss is unchanged over the same window — differences in the 4th–5th decimal with no
systematic drift — confirming the fix alters only the clipping scale, not the
mathematics:

```
Lumen current: 12.60278 12.59553 12.58319 12.58988 12.58122
Lumen fixed:   12.60291 12.59596 12.58199 12.58854 12.58262
```

### DeepSeek-V3 on Megatron's own MLA

Megatron `pretrain_gpt.py`, `--multi-latent-attention --transformer-impl transformer_engine`,
6 layers (3 dense + 3 MoE, following V3's `first_k_dense_replace=3`), hidden 7168,
128 heads, `q_lora_rank=1536`, `kv_lora_rank=512`, 256 experts, TP=8, EP=8, bf16.
No third-party model code; the only variable is the one-line change to
`Linear.reset_parameters`.

| | it 1 | it 2 | it 3 | it 4 |
|---|---|---|---|---|
| stock TE | 12.228 | 12.048 | 11.886 | 11.815 |
| fixed TE | 9.011 | 8.848 | 8.739 | 8.656 |
| ratio | 1.3570 | 1.3617 | 1.3601 | 1.3649 |

Loss is again unchanged (13.19588 vs 13.19578, 13.20511 vs 13.20663, ...).

## History

Introduced in `044903374` (2023-02-10, "QKV parameters unfused path fixes and
optimization" #66), which added weight marking to `transformer_engine/pytorch/module.py`.
Verified by bisect: the preceding commit `78b4e9339` (2023-02-07) has no weight-marking
call.

The initial code drop (`996ea169c`, 2022-09-27) marked **only biases**, which in
column-parallel layers genuinely are sharded. `parallel_mode: Optional[str] = None` was
already reachable by `5612ba784` (2022-10-04), so the non-parallel case existed when the
unconditional marking landed.

Everything since is refactoring, not semantic change: `c6a4a4e08` (2023-05-09) moved the
code into `module/linear.py`; v1.3 moved it from `__init__` into `reset_parameters`. The
behaviour is unchanged in ~3.5 years and is present in current `main`.

## Fix

Mark only genuinely sharded weights:

```python
is_parallel=self.parallel_mode is not None,
```

This restores TE's own documented default. `LayerNormLinear`, `GroupedLinear` and
`LayerNormMLP` should be checked for the same pattern.

A downstream workaround needing no TE change is to clear the attribute after
construction, before the optimizer is built — the distributed optimizer copies TP
attributes onto its shards via `copy_tensor_model_parallel_attributes`, so the cleared
value propagates:

```python
for param in replicated_module.parameters():
    param.tensor_model_parallel = False
    param.partition_dim = -1
    param.partition_stride = 1
```

## Caveats

* Measured on DeepSeek-V4-Flash and DeepSeek-V3, both at TP=8, DP=1, with Adam. The
  ratios are specific to those configurations; they follow `sqrt(1 + (TP-1)f)` and vary
  with model, layer count and TP size.
* Both runs are depth-reduced (4 and 6 layers) for single-node capacity, so the
  dense/MoE balance differs from production — real DeepSeek-V3 is 58 MoE + 3 dense,
  which changes `f` and hence the ratio. Production TP is typically higher than 8, where
  the scaling law predicts a larger factor.
* Verified that the reported norm and the clipping coefficient change. The
  optimizer-sensitivity table is derived from the update rules, not measured end to end;
  a convergence comparison against a reference curve has not been run.
* The DP>1 path is a code-path inference (attributes propagate via
  `copy_tensor_model_parallel_attributes`), not measured.
