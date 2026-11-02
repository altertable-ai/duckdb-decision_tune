"""Export the text decoder of OneJev1.6-0.8B to ONNX, as a decision model.

OneJev1.6-0.8B (https://huggingface.co/OmniJev/OneJev1.6-0.8B, Apache-2.0) is a full fine-tune of Qwen3.5-0.8B. It
answers a question by the output-head logits of the letters A..Z at the last token of a chat prompt
(`onejev_prompt.render`). The vision tower is not exported. One graph runs a batch of right-padded prompts:

    input_ids  int64 [B, T]   the prompts, each followed by padding up to T
    lengths    int64 [B]      each prompt's length
    -> logits  float32 [B, 26]  the letter logits at each prompt's last token; a softmax over a question's first
                                letters gives its probabilities

The model is causal, so padding after a prompt never reaches it: a batch needs no masks beyond the causal one.

The graph runs the reference weights on ONNX Runtime's own CPU kernels:
- the 18 Gated DeltaNet layers run the delta rule as `com.microsoft.LinearAttention` (update_rule gated_delta),
  with q and k L2-normalized before it; their short causal convolution is four shifted multiply-adds, which ONNX
  Runtime runs faster than `CausalConvWithState` or a depthwise `Conv`;
- the 6 full-attention layers are `com.microsoft.GroupQueryAttention` (8 query heads, 2 key/value heads, causal),
  after their per-head RMSNorm and `com.microsoft.RotaryEmbedding` on the first 64 of the 256 dims of each head;
- each layer's projections are one MatMul (q, k, v, z, b and a; or q, gate, k and v; gate and up of the MLP);
- the RMSNorms, whose weight is stored as an offset from 1, are written so that ONNX Runtime fuses them into
  `SimplifiedLayerNormalization`;
- only each prompt's last token goes through the last layer's output projection and MLP, the final norm and the
  letter rows of the output head.

The released graph runs the 96 projections as `com.microsoft.DynamicQuantizeMatMul` (int8 weights, activations
quantized at run time), about three times the fp32 GEMM's speed on arm64, and keeps the embedding in fp16
(`store_int8`). On 216 multilingual support-ticket decisions it gives the fp32 answer on 212; keeping the MLP down
projections in fp32 gave 214 for about 13% more time.

Both graphs are checked against the unmodified transformers model (PyTorch, fp32), prompt by prompt. Outputs land in
`out/onejev/`: model_fp32.onnx (with its external data) and model.onnx.

With --custom-ops, it also writes graphs on the extension's own CPU kernels (src/custom_ops.cpp), which Python's
onnxruntime loads from the library custom_ops/build.sh builds:
- each Gated DeltaNet layer, between its input and output projections, is one `decide_onnx.GatedDeltaNetMixer` node
  (src/gated_delta_net.cpp), which reads the input projection in place, runs the conv, norms and gates in one pass,
  the delta rule in chunks of 64 tokens on GEMMs, and the gated norm. ONNX Runtime's CPU LinearAttention runs the
  delta rule token by token;
- each attention layer's attention is one `decide_onnx.CausalAttention` node (src/causal_attention.cpp), in blocks of
  64 queries on GEMMs. GroupQueryAttention's CPU flash-attention path takes about 4 times as long (ONNX Runtime
  1.29), and its other path holds every head's T x T scores;
- each MLP's activation is one `decide_onnx.SwiGlu` node (src/elementwise.cpp), which reads the fused gate and up
  projection in place;
- each residual connection and the RMSNorm after it are one `decide_onnx.AddRmsNorm` node (src/elementwise.cpp).

Usage: uv run python -I export_onejev.py [--out out/onejev] [--skip-export] [--custom-ops out/libdecide_onnx_ops.dylib]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` leaves the script's directory out
import onejev_prompt  # noqa: E402
from export_onnx import contrib_op  # noqa: E402

_S = onnx.defs.OpSchema

HF_REPO = "OmniJev/OneJev1.6-0.8B"
HF_REVISION = "c89933c277629b707895bff752c61d2b0664c85e"
HF_FILES = ["*.json", "*.jinja", "model.safetensors"]
INPUT_NAMES = ["input_ids", "lengths"]
OUTPUT_NAMES = ["logits"]
SLOTS = len(onejev_prompt.LETTERS)
VERIFY_TOLERANCE = 2e-3
VERIFY_INT8_TOLERANCE = 0.5  # observed worst case: about 0.3
# A prompt in a batch against the same prompt alone, same graph: float rounding only.
VERIFY_BATCH_DRIFT = 1e-3

ROTARY_EMBEDDING = contrib_op("RotaryEmbedding",
                              {"input": "T", "position_ids": "M", "cos_cache": "T", "sin_cache": "T"},
                              {"num_heads": _S.AttrType.INT, "rotary_embedding_dim": _S.AttrType.INT})
GROUP_QUERY_ATTENTION = contrib_op("GroupQueryAttention",
                                   {"query": "T", "key": "T", "value": "T", "past_key": "T?", "past_value": "T?",
                                    "seqlens_k": "I", "total_sequence_length": "I"},
                                   {"num_heads": _S.AttrType.INT, "kv_num_heads": _S.AttrType.INT,
                                    "scale": _S.AttrType.FLOAT},
                                   outputs=("output", "present_key", "present_value"))
LINEAR_ATTENTION = contrib_op("LinearAttention",
                              {"query": "T", "key": "T", "value": "T", "past_state": "T?", "decay": "T", "beta": "T"},
                              {"q_num_heads": _S.AttrType.INT, "kv_num_heads": _S.AttrType.INT,
                               "update_rule": _S.AttrType.STRING},
                              outputs=("output", "present_state"))
# The extension's fused Gated DeltaNet token mixer (src/include/decide_onnx/gated_delta_net.hpp).
CUSTOM_DOMAIN = "decide_onnx"
GATED_DELTA_NET_MIXER = contrib_op("GatedDeltaNetMixer",
                                   {"projection": "T", "taps": "T", "neg_a": "T", "dt_bias": "T", "norm_weight": "T"},
                                   {}, domain=CUSTOM_DOMAIN)
# The extension's causal attention (src/include/decide_onnx/causal_attention.hpp).
CAUSAL_ATTENTION = contrib_op("CausalAttention", {"query": "T", "key": "T", "value": "T"}, {}, domain=CUSTOM_DOMAIN)
# The extension's MLP activation and residual norm (src/include/decide_onnx/elementwise.hpp).
SWIGLU = contrib_op("SwiGlu", {"x": "T"}, {}, domain=CUSTOM_DOMAIN)
ADD_RMS_NORM = contrib_op("AddRmsNorm", {"x": "T", "delta": "T", "weight": "T"}, {}, outputs=("sum", "normed"),
                          domain=CUSTOM_DOMAIN)
# The epsilon of the RMSNorms that the custom ops run (src/custom_ops.cpp).
CUSTOM_OPS_NORM_EPS = 1e-6


def load_reference():
    """The published model (CPU, fp32), its tokenizer and its snapshot directory."""
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    path = Path(snapshot_download(HF_REPO, revision=HF_REVISION, allow_patterns=HF_FILES))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.float32).eval()
    return model, AutoTokenizer.from_pretrained(path), path


def slot_ids(tokenizer) -> list[int]:
    ids = [tokenizer.encode(c, add_special_tokens=False) for c in onejev_prompt.LETTERS]
    if any(len(i) != 1 for i in ids):
        raise SystemExit("a letter A..Z is not one token")
    return [i[0] for i in ids]


def rms_norm(x, weight, eps):
    # Written as x / sqrt(...) so that ONNX Runtime fuses it into SimplifiedLayerNormalization.
    return weight * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps))


@torch.library.custom_op("onejev::rotary_embedding", mutates_args=())
def rotary_embedding(x: torch.Tensor, position_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     num_heads: int, rotary_dim: int) -> torch.Tensor:
    """RoPE with rotate_half on the first `rotary_dim` dims of each head of [B, T, heads * dim]; cos and sin are
    [T, rotary_dim / 2]. Exported as com.microsoft.RotaryEmbedding (non-interleaved)."""
    heads = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    rot, rest = heads[..., :rotary_dim], heads[..., rotary_dim:]
    c, s = cos[position_ids][:, :, None, :], sin[position_ids][:, :, None, :]
    x1, x2 = rot.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, rest), dim=-1).reshape(x.shape)


@rotary_embedding.register_fake
def _(x, position_ids, cos, sin, num_heads, rotary_dim):
    return torch.empty_like(x)


def onnx_rotary_embedding(x, position_ids, cos, sin, num_heads: int, rotary_dim: int):
    return ROTARY_EMBEDDING(x, position_ids, cos, sin, num_heads=num_heads, rotary_embedding_dim=rotary_dim)


@torch.library.custom_op("onejev::causal_attention", mutates_args=())
def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlens_k: torch.Tensor,
                     total_length: torch.Tensor, num_heads: int, kv_num_heads: int, scale: float) -> torch.Tensor:
    """Causal grouped-query attention over [B, T, heads * dim] inputs. Exported as com.microsoft.GroupQueryAttention,
    which takes each row's length - 1 as `seqlens_k` and T as `total_length` (int32)."""
    split = lambda x, h: x.reshape(x.shape[0], x.shape[1], h, -1).transpose(1, 2)
    out = F.scaled_dot_product_attention(split(q, num_heads), split(k, kv_num_heads), split(v, kv_num_heads),
                                         is_causal=True, scale=scale, enable_gqa=True)
    return out.transpose(1, 2).reshape(q.shape)


@causal_attention.register_fake
def _(q, k, v, seqlens_k, total_length, num_heads, kv_num_heads, scale):
    return torch.empty_like(q)


def onnx_causal_attention(q, k, v, seqlens_k, total_length, num_heads: int, kv_num_heads: int, scale: float):
    # The present key and value (the cache) are not used.
    return GROUP_QUERY_ATTENTION(q, k, v, None, None, seqlens_k, total_length, num_heads=num_heads,
                                 kv_num_heads=kv_num_heads, scale=scale)[0]


@torch.library.custom_op("onejev::blocked_causal_attention", mutates_args=())
def blocked_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal grouped-query attention over [B, T, heads, dim] q and [B, T, kv_heads, dim] k and v, scaled by
    dim^-0.5. Exported as decide_onnx.CausalAttention, the extension's CPU kernel."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                         enable_gqa=True)
    return out.transpose(1, 2).contiguous()


@blocked_causal_attention.register_fake
def _(q, k, v):
    return torch.empty_like(q)


def onnx_blocked_causal_attention(q, k, v):
    return CAUSAL_ATTENTION(q, k, v)


@torch.library.custom_op("onejev::gated_delta_rule", mutates_args=())
def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor,
                     num_heads: int) -> torch.Tensor:
    """The gated delta rule over [B, T, heads * dim] q (L2-normalized), k (L2-normalized) and v, with [B, T, heads]
    log-space decay and beta; queries are scaled by dim^-0.5. Exported as com.microsoft.LinearAttention."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    split = lambda x: x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    out, _ = torch_chunk_gated_delta_rule(split(q), split(k), split(v), g=decay, beta=beta)
    return out.reshape(v.shape)


@gated_delta_rule.register_fake
def _(q, k, v, decay, beta, num_heads):
    return torch.empty_like(v)


def onnx_gated_delta_rule(q, k, v, decay, beta, num_heads: int):
    # The final state is not used.
    return LINEAR_ATTENTION(q, k, v, None, decay, beta, q_num_heads=num_heads, kv_num_heads=num_heads,
                            update_rule="gated_delta")[0]


def l2_normalize(x, heads, eps=1e-6):
    """transformers' l2norm on each head of [B, T, heads * dim]."""
    h = x.reshape(x.shape[0], x.shape[1], heads, -1)
    return (h * torch.rsqrt((h * h).sum(-1, keepdim=True) + eps)).reshape(x.shape)


def causal_conv_silu(x, taps):
    """The depthwise causal convolution of [B, T, C] with taps [K, C] (taps[-1] on the current token), then SiLU."""
    K, T = taps.shape[0], x.shape[1]
    padded = F.pad(x, (0, 0, K - 1, 0))
    y = padded[:, 0:T] * taps[0]
    for j in range(1, K):
        y = y + padded[:, j:j + T] * taps[j]
    return F.silu(y)


def mixer(mixed_qkv, a, b, taps, neg_a, dt_bias, heads: int):
    """A Gated DeltaNet layer between its input projection and its gated norm: the causal conv and SiLU of the
    [B, T, C] projection, the L2-normalized q and k, the decay and beta gates, and the delta rule -> [B, T, H * Dv]."""
    key_dim = value_dim = mixed_qkv.shape[-1] // (3 * heads) * heads
    q, k, v = causal_conv_silu(mixed_qkv, taps).split([key_dim, key_dim, value_dim], dim=-1)
    decay = neg_a * F.softplus(a + dt_bias)
    return gated_delta_rule(l2_normalize(q, heads), l2_normalize(k, heads), v, decay, b.sigmoid(), heads)


@torch.library.custom_op("onejev::gated_delta_net_mixer", mutates_args=())
def gated_delta_net_mixer(projection: torch.Tensor, taps: torch.Tensor, neg_a: torch.Tensor, dt_bias: torch.Tensor,
                          norm_weight: torch.Tensor) -> torch.Tensor:
    """A Gated DeltaNet layer between its input and output projections, as one op: the [B, T, C + H * D + 2 * H]
    projection holds mixed_qkv, z, b and a. Exported as decide_onnx.GatedDeltaNetMixer, the extension's CPU kernel: the
    conv, norms and gates in one pass, the delta rule in chunks of 64 tokens on GEMMs, and the gated norm."""
    heads, dim = neg_a.shape[0], norm_weight.shape[0]
    qkv, z, b, a = projection.split([3 * heads * dim, heads * dim, heads, heads], dim=-1)
    batch, length = projection.shape[:2]
    out = rms_norm(mixer(qkv, a, b, taps, neg_a, dt_bias, heads).reshape(batch, length, heads, dim), norm_weight,
                   CUSTOM_OPS_NORM_EPS)
    return (out * F.silu(z.reshape(batch, length, heads, dim))).reshape(batch, length, heads * dim)


@gated_delta_net_mixer.register_fake
def _(projection, taps, neg_a, dt_bias, norm_weight):
    return projection.new_empty(projection.shape[0], projection.shape[1], neg_a.shape[0] * norm_weight.shape[0])


def onnx_gated_delta_net_mixer(projection, taps, neg_a, dt_bias, norm_weight):
    return GATED_DELTA_NET_MIXER(projection, taps, neg_a, dt_bias, norm_weight)


@torch.library.custom_op("onejev::swiglu", mutates_args=())
def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SiLU of the first half of the last dim times the second. Exported as decide_onnx.SwiGlu."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


@swiglu.register_fake
def _(x):
    return x.new_empty(*x.shape[:-1], x.shape[-1] // 2)


def onnx_swiglu(x):
    return SWIGLU(x)


@torch.library.custom_op("onejev::add_rms_norm", mutates_args=())
def add_rms_norm(x: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x + delta, and its RMSNorm. Exported as decide_onnx.AddRmsNorm."""
    total = x + delta
    return total, rms_norm(total, weight, CUSTOM_OPS_NORM_EPS)


@add_rms_norm.register_fake
def _(x, delta, weight):
    return torch.empty_like(x), torch.empty_like(x)


def onnx_add_rms_norm(x, delta, weight):
    return ADD_RMS_NORM(x, delta, weight)


class Decoder(torch.nn.Module):
    """OneJev's text decoder and letter readout on concatenated copies of the reference weights."""

    def __init__(self, model, slots: list[int], custom_ops: bool = False):
        super().__init__()
        self.custom_ops = custom_ops
        lm = model.model.language_model
        cfg = self.config = lm.config
        self.eps = cfg.rms_norm_eps
        assert not custom_ops or self.eps == CUSTOM_OPS_NORM_EPS
        self.embed = lm.embed_tokens.weight.detach()
        self.slot_rows = model.get_output_embeddings().weight.detach()[slots].T.contiguous()  # [hidden, 26]
        self.final_norm = 1.0 + lm.norm.weight.detach()
        self.heads, self.kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.rotary_dim = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
        theta = cfg.rope_parameters["rope_theta"]
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        self.layers = []
        for layer in lm.layers:
            mlp = layer.mlp
            spec = {"type": layer.block_type,
                    "in_norm": 1.0 + layer.input_layernorm.weight.detach(),
                    "post_norm": 1.0 + layer.post_attention_layernorm.weight.detach(),
                    "gate_up": torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight]).detach().T.contiguous(),
                    "down": mlp.down_proj.weight.detach().T.contiguous()}
            if layer.block_type == "linear_attention":
                la = layer.linear_attn
                spec.update(
                    heads=la.num_v_heads, key_dim=la.key_dim, value_dim=la.value_dim,
                    proj=torch.cat([la.in_proj_qkv.weight, la.in_proj_z.weight, la.in_proj_b.weight,
                                    la.in_proj_a.weight]).detach().T.contiguous(),
                    taps=la.conv1d.weight.detach()[:, 0, :].T.contiguous(),  # [K, C]
                    neg_a=-la.A_log.detach().float().exp(), dt_bias=la.dt_bias.detach(),
                    out_norm=la.norm.weight.detach(), out=la.out_proj.weight.detach().T.contiguous())
            else:
                at = layer.self_attn
                # q_proj interleaves each head's query and gate: [q_0, gate_0, q_1, gate_1, ...]. Put the queries
                # first so that one Split separates them.
                q_gate = at.q_proj.weight.detach().reshape(self.heads, 2, self.head_dim, -1)
                spec.update(
                    proj=torch.cat([q_gate[:, 0].reshape(-1, q_gate.shape[-1]), q_gate[:, 1].reshape(-1, q_gate.shape[-1]),
                                    at.k_proj.weight.detach(), at.v_proj.weight.detach()]).T.contiguous(),
                    q_norm=1.0 + at.q_norm.weight.detach(), k_norm=1.0 + at.k_norm.weight.detach(),
                    scale=at.scaling, out=at.o_proj.weight.detach().T.contiguous())
                # decide_onnx.CausalAttention scales by head_dim^-0.5, as Qwen3.5 does.
                assert not custom_ops or abs(at.scaling - self.head_dim ** -0.5) < 1e-9
            self.layers.append(spec)
        # Register every tensor so that the export stores them as initializers.
        for i, spec in enumerate(self.layers):
            for name, value in spec.items():
                if isinstance(value, torch.Tensor):
                    self.register_buffer(f"l{i}_{name}", value, persistent=False)
                    spec[name] = getattr(self, f"l{i}_{name}")
        for name in ("embed", "slot_rows", "final_norm", "inv_freq"):
            value = getattr(self, name)
            delattr(self, name)
            self.register_buffer(name, value, persistent=False)

    def forward(self, input_ids, lengths):
        batch, length = input_ids.shape
        h = F.embedding(input_ids, self.embed)
        position_ids = torch.arange(length)[None].expand(batch, length)
        freqs = torch.arange(length, dtype=torch.float32)[:, None] * self.inv_freq[None, :]
        cos, sin = freqs.cos(), freqs.sin()
        seqlens_k = (lengths - 1).to(torch.int32)
        total_length = torch.tensor(length, dtype=torch.int32)
        last = lengths - 1
        rows = torch.arange(batch)
        delta = None  # the previous layer's MLP output, which the next norm adds to h
        for i, spec in enumerate(self.layers):
            final = i == len(self.layers) - 1
            if delta is None:
                x = rms_norm(h, spec["in_norm"], self.eps)
            else:
                h, x = self.add_norm(h, delta, spec["in_norm"])
            if spec["type"] == "linear_attention":
                heads, key_dim, value_dim = spec["heads"], spec["key_dim"], spec["value_dim"]
                if self.custom_ops:
                    out = gated_delta_net_mixer(x @ spec["proj"], spec["taps"], spec["neg_a"], spec["dt_bias"],
                                                spec["out_norm"])
                else:
                    qkv, z, b, a = (x @ spec["proj"]).split([2 * key_dim + value_dim, value_dim, heads, heads], dim=-1)
                    out = mixer(qkv, a, b, spec["taps"], spec["neg_a"], spec["dt_bias"], heads)
                    out = rms_norm(out.reshape(batch, length, heads, -1), spec["out_norm"], self.eps)
                    out = (out * F.silu(z.reshape(batch, length, heads, -1))).reshape(batch, length, -1)
            else:
                hd = self.head_dim
                q, gate, k, v = (x @ spec["proj"]).split(
                    [self.heads * hd, self.heads * hd, self.kv_heads * hd, self.kv_heads * hd], dim=-1)
                q = rms_norm(q.reshape(batch, length, self.heads, hd), spec["q_norm"], self.eps).reshape(q.shape)
                k = rms_norm(k.reshape(batch, length, self.kv_heads, hd), spec["k_norm"], self.eps).reshape(k.shape)
                q = rotary_embedding(q, position_ids, cos, sin, self.heads, self.rotary_dim)
                k = rotary_embedding(k, position_ids, cos, sin, self.kv_heads, self.rotary_dim)
                if self.custom_ops:
                    split = lambda x, h: x.reshape(batch, length, h, hd)
                    out = blocked_causal_attention(split(q, self.heads), split(k, self.kv_heads),
                                                   split(v, self.kv_heads)).reshape(q.shape)
                else:
                    out = causal_attention(q, k, v, seqlens_k, total_length, self.heads, self.kv_heads,
                                           spec["scale"])
                out = out * gate.sigmoid()
            if final:
                # Only each prompt's last token is read: the rest of the last layer runs on those rows alone.
                h, out = h[rows, last], out[rows, last]
            h, x = self.add_norm(h, out @ spec["out"], spec["post_norm"])
            if self.custom_ops:
                act = swiglu(x @ spec["gate_up"])
            else:
                gate, up = (x @ spec["gate_up"]).chunk(2, dim=-1)
                act = F.silu(gate) * up
            delta = act @ spec["down"]
        return self.add_norm(h, delta, self.final_norm)[1] @ self.slot_rows

    def add_norm(self, h, delta, weight):
        """h + delta, and its RMSNorm."""
        if self.custom_ops:
            return add_rms_norm(h, delta, weight)
        h = h + delta
        return h, rms_norm(h, weight, self.eps)


def export_graph(model, slots, path: Path, custom_ops: bool = False) -> None:
    decoder = Decoder(model, slots, custom_ops).eval()
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    length_dim = torch.export.Dim("length", min=2, max=262144)
    example = (torch.randint(100, 40000, (2, 70)), torch.tensor([70, 50]))
    with torch.inference_mode():
        program = torch.onnx.export(
            decoder, example, dynamo=True, opset_version=21, input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            custom_translation_table={torch.ops.onejev.rotary_embedding.default: onnx_rotary_embedding,
                                      torch.ops.onejev.causal_attention.default: onnx_causal_attention,
                                      torch.ops.onejev.blocked_causal_attention.default: onnx_blocked_causal_attention,
                                      torch.ops.onejev.gated_delta_rule.default: onnx_gated_delta_rule,
                                      torch.ops.onejev.gated_delta_net_mixer.default: onnx_gated_delta_net_mixer,
                                      torch.ops.onejev.swiglu.default: onnx_swiglu,
                                      torch.ops.onejev.add_rms_norm.default: onnx_add_rms_norm},
            dynamic_shapes={"input_ids": {0: batch_dim, 1: length_dim}, "lengths": {0: batch_dim}},
            external_data=True,
        )
    program.save(str(path), external_data=True)


VERIFY_STATES = [
    "The order arrived broken. I want my money back.",
    {"subject": "Double charge", "message": "I was charged twice this month."},
    "Le colis est arrivé cassé, je veux être remboursé.",
    "注文してから1時間後にキャンセルしたのに、料金が請求されました。",
    "a",
]
VERIFY_QUESTIONS = [
    {"type": "choice", "instructions": "Which team should handle this customer message?",
     "criteria": {"billing": "Billing: charges, refunds, invoices", "shipping": "Shipping: delivery, lost or damaged packages",
                  "tech": "Tech support: bugs, crashes, login problems"}},
    {"type": "noul", "instructions": "Is the customer asking for a refund?"},
]
# Lengths of random token prompts, around the 64-token chunks of the delta rule and longer.
VERIFY_RANDOM_LENGTHS = [2, 63, 64, 65, 129, 700, 3000]


def verify_prompts(tokenizer):
    prompts = [tokenizer.encode(onejev_prompt.render(tokenizer, s, q)[0], add_special_tokens=False)
               for s in VERIFY_STATES for q in VERIFY_QUESTIONS]
    g = torch.Generator().manual_seed(0)
    prompts += [torch.randint(0, 240000, (n,), generator=g).tolist() for n in VERIFY_RANDOM_LENGTHS]
    return prompts


def reference_logits(model, slots, ids):
    with torch.inference_mode():
        h = model.model.language_model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0, -1]
        return (model.get_output_embeddings().weight[slots] @ h).numpy()


def batch_feed(prompts, pad_id):
    length = max(len(p) for p in prompts)
    ids = np.full((len(prompts), length), pad_id, np.int64)
    for row, p in enumerate(prompts):
        ids[row, :len(p)] = p
    return {"input_ids": ids, "lengths": np.array([len(p) for p in prompts], np.int64)}


def make_session(path: Path, threads: int = 0, custom_ops: Path | None = None):
    """A CPU session; `custom_ops` is the library of the extension's ops (custom_ops/build.sh), for fused graphs."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.add_session_config_entry("session.disable_quant_qdq", "1")
    if custom_ops is not None:
        options.register_custom_ops_library(str(custom_ops))
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def verify_graph(model, tokenizer, slots, path: Path, tolerance: float, custom_ops: Path | None = None) -> None:
    """Compare ONNX Runtime with the unmodified transformers model, prompt by prompt; then the short prompts as one
    right-padded batch against each alone. Raise on mismatch."""
    session = make_session(path, custom_ops=custom_ops)
    prompts = verify_prompts(tokenizer)
    alone = []
    for ids in prompts:
        expected = reference_logits(model, slots, ids)
        logits = session.run(None, batch_feed([ids], tokenizer.pad_token_id))[0][0]
        alone.append(logits)
        err = float(np.abs(logits - expected).max())
        print(f"  verify T={len(ids)}: max abs logit err {err:.3e}")
        if not err <= tolerance:
            raise SystemExit(f"verification failed for {path.name}: T={len(ids)}")
    short = [i for i, p in enumerate(prompts) if len(p) <= 256]
    logits = session.run(None, batch_feed([prompts[i] for i in short], tokenizer.pad_token_id))[0]
    drift = float(np.abs(logits - np.stack([alone[i] for i in short])).max())
    print(f"  verify batch of {len(short)}: {drift:.3e} from each prompt alone")
    if not drift <= VERIFY_BATCH_DRIFT:
        raise SystemExit(f"verification failed for {path.name}: batch")


def store_int8(src: Path, dst: Path, keep_fp32_inputs: tuple[int, ...] = ()) -> None:
    """Rewrite the fp32 graph to run its projections in int8 and to keep its embedding in fp16.
    - The 96 projections (the fused input projection, the output projection, gate_up and down of 24 layers) become
      `com.microsoft.DynamicQuantizeMatMul`: int8 weights, symmetric, one scale per output column; ONNX Runtime
      quantizes the activations per tensor at run time and multiplies in int8 (KleidiAI on arm64), about three times
      the fp32 GEMM's speed. The weights stay int8 in memory.
    - The token embedding is stored in fp16, and only the gathered rows are cast to fp32, so the table stays fp16 in
      memory too.
    - The letter rows of the output head and the small tensors stay fp32, as do the projections whose input size is
      in `keep_fp32_inputs` (3584: the MLP down projections)."""
    model = onnx.load(str(src))
    graph = model.graph
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    inits = {i.name: i for i in graph.initializer}
    replaced, nodes, projections, kept = set(), [], 0, 0
    added = []
    for node in graph.node:
        weight = inits.get(node.input[1]) if len(node.input) > 1 else None
        if (node.op_type == "MatMul" and weight is not None and weight.data_type == onnx.TensorProto.FLOAT
                and len(weight.dims) == 2 and min(weight.dims) >= 1024):
            if weight.dims[0] in keep_fp32_inputs:
                nodes.append(node)
                kept += 1
                continue
            w = onnx.numpy_helper.to_array(weight)
            scale = (np.abs(w).max(0) / 127).astype(np.float32)
            scale[scale == 0] = 1
            q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
            base = weight.name
            added += [onnx.numpy_helper.from_array(q, base + "__int8"),
                      onnx.numpy_helper.from_array(scale, base + "__scale"),
                      onnx.numpy_helper.from_array(np.zeros_like(q[0]), base + "__zero")]
            nodes.append(onnx.helper.make_node("DynamicQuantizeMatMul",
                                               [node.input[0], base + "__int8", base + "__scale", base + "__zero"],
                                               list(node.output), name=node.name, domain="com.microsoft"))
            replaced.add(base)
            projections += 1
        elif (node.op_type == "Gather" and node.input[0] in inits
              and inits[node.input[0]].data_type == onnx.TensorProto.FLOAT and len(inits[node.input[0]].dims) == 2
              and inits[node.input[0]].dims[0] > 100000):
            table = inits[node.input[0]]
            added.append(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(table).astype(np.float16),
                                                      table.name + "__fp16"))
            rows = node.output[0] + "__fp16"
            nodes += [onnx.helper.make_node("Gather", [table.name + "__fp16", node.input[1]], [rows],
                                            name=node.name, axis=0),
                      onnx.helper.make_node("Cast", [rows], [node.output[0]], to=onnx.TensorProto.FLOAT)]
            replaced.add(table.name)
        else:
            nodes.append(node)
    if projections + kept != 96:
        raise SystemExit(f"expected the 96 projections, found {projections + kept}")
    keep = [i for i in graph.initializer if i.name not in replaced] + added
    del graph.initializer[:]
    graph.initializer.extend(keep)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.save_model(model, str(dst))


def benchmark(path: Path, tokenizer, custom_ops: Path | None = None) -> None:
    """One thread: one prompt alone, and the ten verification chat prompts as one batch."""
    session = make_session(path, threads=1, custom_ops=custom_ops)
    prompts = verify_prompts(tokenizer)[:len(VERIFY_STATES) * len(VERIFY_QUESTIONS)]
    for label, feed in [(f"one prompt, T={len(prompts[0])}", batch_feed(prompts[:1], tokenizer.pad_token_id)),
                        (f"batch of {len(prompts)}, T={max(map(len, prompts))}", batch_feed(prompts, tokenizer.pad_token_id))]:
        session.run(None, feed)
        n, start = 5, time.perf_counter()
        for _ in range(n):
            session.run(None, feed)
        ms = (time.perf_counter() - start) / n * 1000
        print(f"  1 thread, {label}: {ms:.1f} ms ({ms / len(feed['lengths']):.1f} ms per prompt)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "onejev")
    parser.add_argument("--skip-export", action="store_true", help="reuse model_fp32.onnx")
    parser.add_argument("--custom-ops", type=Path, metavar="LIBRARY",
                        help="also write model_fused_fp32.onnx and model_fused.onnx, which run the extension's "
                             "decide_onnx.GatedDeltaNetMixer and decide_onnx.CausalAttention, and check them with "
                             "LIBRARY (out/libdecide_onnx_ops.*, from custom_ops/build.sh)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"exporting {HF_REPO}@{HF_REVISION}")
    model, tokenizer, _ = load_reference()
    slots = slot_ids(tokenizer)
    fp32_path = args.out / "model_fp32.onnx"
    if not args.skip_export:
        export_graph(model, slots, fp32_path)
    verify_graph(model, tokenizer, slots, fp32_path, VERIFY_TOLERANCE)
    benchmark(fp32_path, tokenizer)
    print("storing the projections as int8")
    onnx_path = args.out / "model.onnx"
    store_int8(fp32_path, onnx_path)
    verify_graph(model, tokenizer, slots, onnx_path, VERIFY_INT8_TOLERANCE)
    benchmark(onnx_path, tokenizer)
    if args.custom_ops:
        print("exporting on the extension's custom ops")
        fused_fp32, fused = args.out / "model_fused_fp32.onnx", args.out / "model_fused.onnx"
        if not args.skip_export:
            export_graph(model, slots, fused_fp32, custom_ops=True)
        verify_graph(model, tokenizer, slots, fused_fp32, VERIFY_TOLERANCE, args.custom_ops)
        store_int8(fused_fp32, fused)
        verify_graph(model, tokenizer, slots, fused, VERIFY_INT8_TOLERANCE, args.custom_ops)
        benchmark(fused, tokenizer, args.custom_ops)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
