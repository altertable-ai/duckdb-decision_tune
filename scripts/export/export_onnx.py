"""Export the vela3-fast decision backbone to ONNX for the decide_onnx DuckDB extension.

vela3-fast (https://huggingface.co/vllm-sr/vela3-fast, Apache-2.0) answers choice and yes/no questions with its
decision backbone: an UltraBERT-380M encoder and a small readout head (`vela3_encoder.DecisionEncoder`). One prompt
is one question:

    [BOS] [Q] question [O] option_1 [O] option_2 ... [SEP_SCHEMA] [SEG_user] state [EOS]

The encoder and the readout become one ONNX graph, for one prompt or a pack of short prompts laid end to end:

    input_ids     int64 [1, T]   the prompts
    position_ids  int64 [1, T]   each token's position within its prompt
    segment_ids   int64 [1, T]   each token's prompt: 0, 1, ... in order (all 0 for one prompt)
    positions     int64 [K]      the position of the [O] before each option, over all prompts
    questions     int64 [K]      the position of the [Q] of each option's prompt
    state_starts  int64 [K]      the position of the [SEG_user] of each option's prompt
    state_ends    int64 [K]      the position of the final [EOS] of each option's prompt
    -> logits     float32 [K]    a softmax over the options of each prompt gives its probabilities

A pack (more than one segment) may only hold prompts of at most PACK_PROMPT_MAX tokens: a sliding window then spans
each whole prompt, so every layer is exact with block-diagonal attention. A pack of short prompts uses the CPU much
better than each prompt alone: the GEMMs run in blocks of 32 rows, and short passes are their least efficient size.

The graph is not a trace of the published modeling code, which builds dense [T, T] attention masks and scores:
O(T^2) memory, about 7 GB at 8K tokens on a CPU. `Encoder` below runs the same weights on ONNX Runtime's own
kernels, with memory linear in T:
- the 8 global layers are `com.microsoft.MultiHeadAttention`. Up to GEMM_ATTENTION_MAX tokens, an `If` gives it an
  all-ones key_padding_mask, which makes ONNX Runtime compute the [heads, T, T] scores with its GEMM (KleidiAI SME on
  arm64): about twice as fast as its flash attention, whose GEMMs are plain NEON, for 512 MB of scores at most. Longer
  prompts take the flash attention, whose memory is linear in T;
- the 16 sliding-window layers attend in blocks of 128 queries, built from slices of the padded keys; a prompt of
  at most 129 tokens fits in one window, so an `If` sends it to the GEMM MultiHeadAttention instead;
- in a pack, every layer is the GEMM MultiHeadAttention with an attention_bias that keeps each prompt to its own
  keys; a prompt's logits match its own pass up to float rounding;
- RoPE is `com.microsoft.RotaryEmbedding`, so q, k and v stay [1, T, hidden] with no transposes in global layers;
- RMSNorm is written so that ONNX Runtime fuses it into `SimplifiedLayerNormalization`.

The released graph stores the weights compactly: the encoder projections as int8 (`DequantizeLinear`, blocks of 128)
and the token embedding as fp16. With `session.disable_quant_qdq = 1`, ONNX Runtime folds them back into fp32 weights
when it loads the model, so inference runs the fp32 kernels from a file a third of the size. Both the fp32 graph and
the released one are checked against the unmodified `DecisionEncoder` (PyTorch, fp32).

Outputs land in `out/`: model_fp32.onnx, and the release: model.onnx, tokenizer.json, manifest.json and the weights'
LICENSE. The extension's `src/include/decide_onnx/model_manifest.hpp` is regenerated.

Usage: uv run python -I export_onnx.py [--out out] [--skip-export]
  --skip-export   reuse out/model_fp32.onnx (verified again) instead of exporting it anew; exports are not
                  byte-identical, so this rebuilds the release of an earlier run
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

_S = onnx.defs.OpSchema


def contrib_op(name, inputs, attributes):
    """An ONNX Runtime contrib op, which `onnx` has no schema for: the part of its schema the export uses.
    `inputs` maps each input to its type ("T": float, "M": int64, "I": int32); a trailing "?" makes it optional."""
    def parameter(n, t):
        if t.endswith("?"):
            return _S.FormalParameter(n, t[:-1], "", param_option=_S.FormalParameterOption.Optional)
        return _S.FormalParameter(n, t)

    schema = _S(name, "com.microsoft", 1,
                inputs=[parameter(n, t) for n, t in inputs.items()],
                outputs=[_S.FormalParameter("output", "T")],
                type_constraints=[("T", ["tensor(float)"], ""), ("M", ["tensor(int64)"], ""),
                                  ("I", ["tensor(int32)"], "")],
                attributes=[_S.Attribute(n, t, "") for n, t in attributes.items()])
    return onnxscript.values.Op(onnxscript.values.Opset("com.microsoft", 1), name, schema)


MULTI_HEAD_ATTENTION = contrib_op("MultiHeadAttention",
                                  {"query": "T", "key": "T", "value": "T", "bias": "T?", "key_padding_mask": "I?",
                                   "attention_bias": "T?"},
                                  {"num_heads": _S.AttrType.INT, "scale": _S.AttrType.FLOAT})
ROTARY_EMBEDDING = contrib_op("RotaryEmbedding",
                              {"input": "T", "position_ids": "M", "cos_cache": "T", "sin_cache": "T"},
                              {"num_heads": _S.AttrType.INT})

ASSET_VERSION = "v5"
HF_REPO = "vllm-sr/vela3-fast"
HF_REVISION = "078c42f90cabdac7964f8e206d95a6f3a1aafacb"
# Everything the decision backbone needs; the span and router adapters are not used.
HF_FILES = ["*.py", "*.json", "model.safetensors", "LICENSE", "SHA256SUMS"]
INPUT_NAMES = ["input_ids", "position_ids", "segment_ids", "positions", "questions", "state_starts", "state_ends"]
OUTPUT_NAMES = ["logits"]
BLOCK = 128
# Longest prompt of a pack: one more than the sliding window.
PACK_PROMPT_MAX = 129
# Longest prompt whose global layers compute the full [heads, T, T] scores (8 heads: 512 MB at 4,096 tokens).
GEMM_ATTENTION_MAX = 4096
# (body tokens, options) checked against PyTorch; a prompt is body + options + 4 tokens. Short, both sides of the
# 129-token switch from global to blocked attention in the sliding-window layers, several blocks, both sides of
# GEMM_ATTENTION_MAX, and long.
VERIFY_CASES = [(3, 2), (40, 3), (123, 2), (124, 2), (125, 4), (127, 5), (300, 3), (1100, 4), (4090, 2), (4091, 2),
                (9000, 2)]
# Packs of VERIFY_CASES-style prompts of at most PACK_PROMPT_MAX tokens, checked against PyTorch and against each
# prompt's own pass.
VERIFY_PACKS = [[(3, 2), (40, 3), (20, 2)], [(123, 2), (3, 4), (60, 3), (30, 2), (2, 2)], [(26, 5)] * 9]
VERIFY_TOLERANCE = 1e-3
# A prompt in a pack against the same prompt alone, same graph: float rounding only.
VERIFY_PACK_DRIFT = 1e-4
# The released graph: int8 encoder projections, symmetric, in blocks of INT8_BLOCK input rows.
INT8_BLOCK = 128
VERIFY_INT8_TOLERANCE = 1e-1  # observed worst case: about 9e-2
REPO_ROOT = Path(__file__).resolve().parents[2]
HEADER_PATH = REPO_ROOT / "src" / "include" / "decide_onnx" / "model_manifest.hpp"


def load_reference():
    """The published `DecisionEncoder` (CPU, fp32) and the snapshot directory it was loaded from."""
    path = Path(snapshot_download(HF_REPO, revision=HF_REVISION, allow_patterns=HF_FILES))
    sys.path.insert(0, str(path))
    from vela3_encoder import DecisionEncoder

    return DecisionEncoder.load(path, device="cpu"), path


def rms_norm(norm, x):
    # Written as x / sqrt(...) so that ONNX Runtime fuses it into SimplifiedLayerNormalization.
    return norm.weight * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon))


@torch.library.custom_op("decide_onnx::rotary_embedding", mutates_args=())
def rotary_embedding(x: torch.Tensor, position_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     num_heads: int) -> torch.Tensor:
    """RoPE with rotate_half on [1, T, heads * dim]; cos and sin are [T, dim / 2]. Exported as
    com.microsoft.RotaryEmbedding (non-interleaved)."""
    heads = x.reshape(1, x.shape[1], num_heads, -1)
    c, s = cos[position_ids][:, :, None, :], sin[position_ids][:, :, None, :]
    x1, x2 = heads.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).reshape(x.shape)


@rotary_embedding.register_fake
def _(x, position_ids, cos, sin, num_heads):
    return torch.empty_like(x)


def onnx_rotary_embedding(x, position_ids, cos, sin, num_heads: int):
    return ROTARY_EMBEDDING(x, position_ids, cos, sin, num_heads=num_heads)


@torch.library.custom_op("decide_onnx::global_attention", mutates_args=())
def global_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, scale: float) -> torch.Tensor:
    """Unmasked attention over [1, T, heads * dim] inputs; exported as com.microsoft.MultiHeadAttention, which runs
    it as a flash attention."""
    split = lambda x: x.reshape(1, x.shape[1], num_heads, -1).transpose(1, 2)
    out = F.scaled_dot_product_attention(split(q), split(k), split(v), scale=scale)
    return out.transpose(1, 2).reshape(q.shape)


@global_attention.register_fake
def _(q, k, v, num_heads, scale):
    return torch.empty_like(q)


def onnx_global_attention(q, k, v, num_heads: int, scale: float):
    return MULTI_HEAD_ATTENTION(q, k, v, num_heads=num_heads, scale=scale)


@torch.library.custom_op("decide_onnx::gemm_attention", mutates_args=())
def gemm_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor, num_heads: int,
                   scale: float) -> torch.Tensor:
    """`global_attention` with `keep`, int32 ones [1, T], as the key_padding_mask of MultiHeadAttention. It masks
    nothing, but any mask makes ONNX Runtime skip its flash attention for the unfused one, whose GEMMs are faster."""
    return global_attention(q, k, v, num_heads, scale)


@gemm_attention.register_fake
def _(q, k, v, keep, num_heads, scale):
    return torch.empty_like(q)


def onnx_gemm_attention(q, k, v, keep, num_heads: int, scale: float):
    return MULTI_HEAD_ATTENTION(q, k, v, None, keep, num_heads=num_heads, scale=scale)


@torch.library.custom_op("decide_onnx::packed_attention", mutates_args=())
def packed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, num_heads: int,
                     scale: float) -> torch.Tensor:
    """`global_attention` plus `bias`, [1, 1, T, T], as the attention_bias of MultiHeadAttention."""
    split = lambda x: x.reshape(1, x.shape[1], num_heads, -1).transpose(1, 2)
    out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=bias, scale=scale)
    return out.transpose(1, 2).reshape(q.shape)


@packed_attention.register_fake
def _(q, k, v, bias, num_heads, scale):
    return torch.empty_like(q)


def onnx_packed_attention(q, k, v, bias, num_heads: int, scale: float):
    return MULTI_HEAD_ATTENTION(q, k, v, None, None, bias, num_heads=num_heads, scale=scale)


def segment_bias(segment_ids, dtype):
    """[1, 1, T, T]: 0 where query and key are in the same prompt, the lowest float elsewhere."""
    same = segment_ids[:, :, None] == segment_ids[:, None, :]
    return torch.where(same, 0.0, torch.finfo(dtype).min).to(dtype)[:, None]


def local_attention(q, k, v, window, scale):
    """Attention over the keys within `window` positions, in blocks of BLOCK queries: O(T * window) memory.
    With window == BLOCK, the keys of block i are the padded blocks i, i + 1 and i + 2: slices, not a gather."""
    assert window == BLOCK, "the sliding window must be one block"
    heads, length, dim = q.shape[1], q.shape[2], q.shape[3]
    count = (length + BLOCK - 1) // BLOCK
    tail = count * BLOCK - length
    span = BLOCK + 2 * window
    starts = torch.arange(count) * BLOCK

    def windows(x):  # [1, H, count, span, D]
        blocks = F.pad(x, (0, 0, window, tail + window)).reshape(1, heads, count + 2, BLOCK, dim)
        return torch.cat((blocks[:, :, :count], blocks[:, :, 1:count + 1], blocks[:, :, 2:]), dim=3)

    k_win, v_win = windows(k), windows(v)
    q_blocks = F.pad(q, (0, 0, 0, tail)).reshape(1, heads, count, BLOCK, dim)
    q_pos = starts[:, None, None] + torch.arange(BLOCK)[None, :, None]
    k_pos = starts[:, None, None] - window + torch.arange(span)[None, None, :]
    allowed = ((q_pos - k_pos).abs() <= window) & (k_pos >= 0) & (k_pos < length)
    scores = torch.matmul(q_blocks, k_win.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    out = torch.matmul(scores.softmax(-1), v_win)  # [1, H, count, BLOCK, D]
    return out.reshape(1, heads, count * BLOCK, dim)[:, :, :length]


def blocked_attention(q, k, v, heads, window, scale):
    """`local_attention` on [1, T, heads * dim] inputs and output."""
    length = q.shape[1]
    q, k, v = (t.reshape(1, length, heads, -1).transpose(1, 2) for t in (q, k, v))
    return local_attention(q, k, v, window, scale).transpose(1, 2).reshape(1, length, -1)


class Encoder(torch.nn.Module):
    """`DecisionEncoder`'s encoder and readout for one unpadded prompt or a pack of them, on the reference's own
    modules."""

    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        self.config = reference.encoder.config
        self.inv_freq = {theta: 1.0 / (theta ** (torch.arange(0, self.config.head_dim, 2, dtype=torch.float32)
                                                 / self.config.head_dim))
                         for theta in set(self.config.rope_theta)}

    def forward(self, input_ids, position_ids, segment_ids, positions, questions, state_starts, state_ends):
        encoder, cfg = self.reference.encoder, self.config
        length = input_ids.shape[1]
        packed = segment_ids[0, -1] > 0
        h = encoder.embed_tokens(input_ids)
        rope = {}
        for theta, inv_freq in self.inv_freq.items():
            freqs = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
            rope[theta] = (freqs.cos(), freqs.sin())
        heads = cfg.num_attention_heads
        keep = torch.ones((1, length), dtype=torch.int32)
        for i, layer in enumerate(encoder.layers):
            attn = layer.self_attn
            x = rms_norm(layer.input_layernorm, h)
            cos, sin = rope[cfg.rope_theta[i]]
            # q, k and v stay [1, T, heads * dim]: what RotaryEmbedding and MultiHeadAttention take.
            q = rotary_embedding(attn.q_proj(x), position_ids, cos, sin, heads)
            k = rotary_embedding(attn.k_proj(x), position_ids, cos, sin, heads)
            v = attn.v_proj(x)
            window = cfg.attention_windows[i]
            if window is None:
                single = lambda q, k, v, keep, scale=attn.scaling: torch.cond(
                    length <= GEMM_ATTENTION_MAX,
                    lambda q, k, v, keep: gemm_attention(q, k, v, keep, heads, scale),
                    lambda q, k, v, keep: global_attention(q, k, v, heads, scale),
                    (q, k, v, keep))
            else:
                # A prompt of at most window + 1 tokens attends everywhere: the global kernel is exact and cheaper.
                single = lambda q, k, v, keep, scale=attn.scaling, window=window: torch.cond(
                    length <= window + 1,
                    lambda q, k, v, keep: gemm_attention(q, k, v, keep, heads, scale),
                    lambda q, k, v, keep: blocked_attention(q, k, v, heads, window, scale),
                    (q, k, v, keep))
            # The bias is built inside the branch: a long single prompt must not pay for a [T, T] tensor.
            out = torch.cond(packed,
                             lambda q, k, v, keep, segment_ids, scale=attn.scaling: packed_attention(
                                 q, k, v, segment_bias(segment_ids, q.dtype), heads, scale),
                             lambda q, k, v, keep, segment_ids: single(q, k, v, keep),
                             (q, k, v, keep, segment_ids))
            h = h + attn.o_proj(out)
            x = rms_norm(layer.post_attention_layernorm, h)
            h = h + layer.mlp(x)
        hidden = rms_norm(encoder.norm, h)[0]  # [T, hidden]
        return self.readout(hidden, positions, questions, state_starts, state_ends)

    def readout(self, hidden, positions, questions, state_starts, state_ends):
        """`DecisionEncoder._logits`, one row per option: the state of its prompt is pooled from [SEG_user] up to
        [EOS]. The pooling is a [K, T] mask product rather than a cumulative sum, which would carry the rounding of
        the earlier prompts of a pack."""
        ref = self.reference
        steps = torch.arange(hidden.shape[0])[None]
        span = (steps >= state_starts[:, None]) & (steps < state_ends[:, None])
        count = (state_ends - state_starts).clamp(min=1).to(hidden.dtype)[:, None]
        pooled = span.to(hidden.dtype) @ hidden / count
        state = F.normalize(ref.w_p(ref.norm(pooled)), dim=-1)
        question = ref.w_q(ref.norm(hidden[questions]))
        options = hidden[positions]
        u = F.normalize(ref.w_o(ref.norm(options)) + question, dim=-1)
        return (u * state).sum(-1) / ref.log_tau.exp() + ref.cls_mlp(options).squeeze(-1)


def export_graph(reference, path: Path) -> None:
    model = Encoder(reference).eval()
    length_dim = torch.export.Dim("length", min=4, max=32768)
    options_dim = torch.export.Dim("options", min=1, max=4096)
    length = 300
    example = (torch.randint(100, 40000, (1, length)), torch.arange(length)[None],
               torch.zeros(1, length, dtype=torch.long), torch.tensor([2, 9, 17]), torch.tensor([1, 1, 1]),
               torch.tensor([30, 30, 30]), torch.tensor([length - 1] * 3))
    with torch.inference_mode():
        program = torch.onnx.export(
            model,
            example,
            dynamo=True,
            opset_version=20,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            custom_translation_table={torch.ops.decide_onnx.global_attention.default: onnx_global_attention,
                                      torch.ops.decide_onnx.gemm_attention.default: onnx_gemm_attention,
                                      torch.ops.decide_onnx.packed_attention.default: onnx_packed_attention,
                                      torch.ops.decide_onnx.rotary_embedding.default: onnx_rotary_embedding},
            dynamic_shapes={"input_ids": {1: length_dim}, "position_ids": {1: length_dim},
                            "segment_ids": {1: length_dim}, "positions": {0: options_dim},
                            "questions": {0: options_dim}, "state_starts": {0: options_dim},
                            "state_ends": {0: options_dim}},
        )
    program.save(str(path))


def verify_prompt(reference, length, options, seed):
    """A random prompt of `length` tokens with the reference's layout, as `DecisionEncoder.encode` builds it."""
    g = torch.Generator().manual_seed(seed)
    mk = reference.mk
    body = torch.randint(200, 49000, (length,), generator=g).tolist()
    ids = [reference.tok.bos_token_id, mk["[Q]"]] + body[:2]
    positions = []
    for i in range(options):
        positions.append(len(ids))
        ids += [mk["[O]"]] + body[2 + i : 3 + i]
    ids.append(mk["[SEP_SCHEMA]"])
    state_start = len(ids)
    ids += [mk["[SEG_user]"]] + body[3 + options :]
    ids.append(reference.tok.eos_token_id)
    return dict(ids=ids, q_pos=1, opt_pos=positions, names=[str(i) for i in range(options)], pool=(state_start, len(ids) - 1))


def store_compact(src: Path, dst: Path) -> None:
    """Rewrite the fp32 graph with its weights stored compactly; the values ONNX Runtime computes from them are fp32.
    - The 168 encoder projections (q, k, v, o, gate, up, down of 24 layers) become int8, symmetric, in blocks of
      INT8_BLOCK along the input dimension, read back by a blocked DequantizeLinear. The scales are those of ONNX
      Runtime's 8-bit MatMulNBits quantizer: the largest weight of a block maps to -128.
    - The token embedding becomes fp16, read back by a Cast. An int8 embedding doubled the error.
    - The readout head and the small tensors stay fp32."""
    model = onnx.load(str(src))
    graph = model.graph
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = max(opset.version, 21)  # DequantizeLinear with block_size
    uses = {}
    for node in graph.node:
        for i, name in enumerate(node.input):
            uses.setdefault(name, set()).add((node.op_type, i))
    keep, nodes, projections = [], [], 0
    for init in graph.initializer:
        name, kind = init.name, uses.get(init.name)
        if init.data_type == onnx.TensorProto.FLOAT and kind == {("MatMul", 1)} and min(init.dims) >= 1024:
            w = onnx.numpy_helper.to_array(init)
            rows, cols = w.shape
            blocks = w.reshape(rows // INT8_BLOCK, INT8_BLOCK, cols)
            peak = np.take_along_axis(blocks, np.abs(blocks).argmax(axis=1)[:, None, :], axis=1)
            scale = (peak / -128).astype(np.float32)
            scale[scale == 0] = 1
            q = np.clip(np.rint(blocks / scale), -128, 127).astype(np.int8)
            keep += [onnx.numpy_helper.from_array(q.reshape(rows, cols), name + "__int8"),
                     onnx.numpy_helper.from_array(scale[:, 0, :], name + "__scale")]
            nodes.append(onnx.helper.make_node("DequantizeLinear", [name + "__int8", name + "__scale"], [name],
                                               axis=0, block_size=INT8_BLOCK))
            projections += 1
        elif init.data_type == onnx.TensorProto.FLOAT and kind == {("Gather", 0)}:
            keep.append(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(init).astype(np.float16), name + "__fp16"))
            nodes.append(onnx.helper.make_node("Cast", [name + "__fp16"], [name], to=onnx.TensorProto.FLOAT))
        else:
            keep.append(init)
    if projections != 168:
        raise SystemExit(f"expected the 168 encoder projections, found {projections}")
    del graph.initializer[:]
    graph.initializer.extend(keep)
    graph_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + graph_nodes)
    onnx.save_model(model, str(dst))


def pack_feed(recs):
    """The graph inputs for `recs` (verify_prompt records) laid end to end."""
    ids, position_ids, segment_ids, positions, questions, state_starts, state_ends = [], [], [], [], [], [], []
    for segment, rec in enumerate(recs):
        start = len(ids)
        ids += rec["ids"]
        position_ids += range(len(rec["ids"]))
        segment_ids += [segment] * len(rec["ids"])
        positions += [start + p for p in rec["opt_pos"]]
        questions += [start + rec["q_pos"]] * len(rec["opt_pos"])
        state_starts += [start + rec["pool"][0]] * len(rec["opt_pos"])
        state_ends += [start + rec["pool"][1]] * len(rec["opt_pos"])
    values = [[ids], [position_ids], [segment_ids], positions, questions, state_starts, state_ends]
    return {name: np.array(value, np.int64) for name, value in zip(INPUT_NAMES, values)}


def reference_logits(reference, rec):
    with torch.inference_mode():
        H = reference.encoder(input_ids=torch.tensor([rec["ids"]]),
                              attention_mask=torch.ones(1, len(rec["ids"]), dtype=torch.long)).last_hidden_state
        return reference._logits(H, [rec])[0].numpy()


def verify_graph(reference, path: Path, tolerance: float) -> None:
    """Compare ONNX Runtime with the unmodified `DecisionEncoder.forward_recs`; raise on mismatch. A pack is also
    compared with each of its prompts run alone."""
    options = ort.SessionOptions()
    # As the extension runs it: fold the int8 weights into fp32 rather than fusing them into MatMulNBits.
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for n, (length, options) in enumerate(VERIFY_CASES):
        rec = verify_prompt(reference, length, options, seed=n)
        expected = reference_logits(reference, rec)
        logits = session.run(None, pack_feed([rec]))[0]
        err = float(np.abs(logits - expected).max())
        print(f"  verify T={len(rec['ids'])} options={options}: max abs logit err {err:.3e}")
        if logits.shape != expected.shape or not err <= tolerance:
            raise SystemExit(f"verification failed for {path.name}: T={len(rec['ids'])}")
    for n, pack in enumerate(VERIFY_PACKS):
        recs = [verify_prompt(reference, length, options, seed=100 * n + i) for i, (length, options) in enumerate(pack)]
        assert all(len(rec["ids"]) <= PACK_PROMPT_MAX for rec in recs)
        logits = session.run(None, pack_feed(recs))[0]
        expected = np.concatenate([reference_logits(reference, rec) for rec in recs])
        alone = np.concatenate([session.run(None, pack_feed([rec]))[0] for rec in recs])
        err, drift = float(np.abs(logits - expected).max()), float(np.abs(logits - alone).max())
        print(f"  verify pack of {len(recs)}, T={sum(len(rec['ids']) for rec in recs)}: max abs logit err {err:.3e}, "
              f"{drift:.3e} from each prompt alone")
        if logits.shape != expected.shape or not err <= tolerance or not drift <= VERIFY_PACK_DRIFT:
            raise SystemExit(f"verification failed for {path.name}: pack {n}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_header(files: list[dict], markers: dict, context_tokens: int) -> None:
    # Laid out exactly as clang-format (DuckDB's .clang-format) formats it, so `make format-check` passes.
    by_name = {f["file_name"]: f for f in files}
    tok, onnx = by_name["tokenizer.json"], by_name["model.onnx"]
    marker_rows = "\n".join(f'static constexpr int64_t {name}_TOKEN = {markers[token]};'
                            for name, token in [("QUESTION", "[Q]"), ("OPTION", "[O]"), ("SCHEMA_SEP", "[SEP_SCHEMA]"),
                                                ("STATE", "[SEG_user]")])
    HEADER_PATH.write_text(f"""// GENERATED by scripts/export/export_onnx.py; do not edit by hand.
#pragma once

#include <cstdint>

namespace duckdb {{
namespace decide_onnx {{

//! A file of the model release, as listed in its `manifest.json`.
struct ModelFile {{
	//! Name in the local cache directory and under the download URL.
	const char *file_name;
	const char *sha256;
	uint64_t file_bytes;
}};

static constexpr const char *MODEL_NAME = "vela3-fast";
static constexpr const char *ASSET_VERSION = "{ASSET_VERSION}";
//! The Hugging Face repository and commit the weights come from (Apache-2.0).
static constexpr const char *HF_REPO = "{HF_REPO}";
static constexpr const char *HF_REVISION = "{HF_REVISION}";

static constexpr ModelFile TOKENIZER_FILE = {{
    "tokenizer.json", "{tok["sha256"]}", {tok["file_bytes"]}ULL}};
static constexpr ModelFile ONNX_FILE = {{
    "model.onnx", "{onnx["sha256"]}", {onnx["file_bytes"]}ULL}};
static constexpr const ModelFile *MODEL_FILES[] = {{&TOKENIZER_FILE, &ONNX_FILE}};

//! The decision markers, appended to the tokenizer's vocabulary (`decision.marker_ids` in config.json).
{marker_rows}

//! `decision.max_len` in config.json: longer prompts are refused, never truncated.
static constexpr uint64_t CONTEXT_TOKENS = {context_tokens};
//! The longest prompt that may share a forward pass with others.
static constexpr uint64_t PACK_PROMPT_TOKENS = {PACK_PROMPT_MAX};

}} // namespace decide_onnx
}} // namespace duckdb
""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--skip-export", action="store_true", help="reuse out/model_fp32.onnx")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"exporting {HF_REPO}@{HF_REVISION}")
    reference, snapshot = load_reference()
    fp32_path, onnx_path = args.out / "model_fp32.onnx", args.out / "model.onnx"
    if not args.skip_export:
        export_graph(reference, fp32_path)
    verify_graph(reference, fp32_path, VERIFY_TOLERANCE)
    print("storing the weights as int8")
    store_compact(fp32_path, onnx_path)
    verify_graph(reference, onnx_path, VERIFY_INT8_TOLERANCE)
    shutil.copyfile(snapshot / "tokenizer.json", args.out / "tokenizer.json")
    shutil.copyfile(snapshot / "LICENSE", args.out / "LICENSE")

    decision = json.loads((snapshot / "config.json").read_text())["decision"]
    files = [{"file_name": name, "sha256": sha256_of(args.out / name), "file_bytes": (args.out / name).stat().st_size}
             for name in ("tokenizer.json", "model.onnx")]
    manifest = {"asset_version": ASSET_VERSION, "hf_repo": HF_REPO, "hf_revision": HF_REVISION,
                "marker_ids": decision["marker_ids"], "context_tokens": decision["max_len"], "files": files}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_header(files, decision["marker_ids"], decision["max_len"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
