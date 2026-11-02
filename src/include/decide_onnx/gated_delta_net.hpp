#pragma once

#include "decide_onnx/cpu_backend.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace decide_onnx {
namespace gated_delta_net {

//! The sizes of one Gated DeltaNet layer over one prompt.
struct Shape {
	int64_t tokens;
	int64_t heads;
	int64_t key_dim;
	int64_t value_dim;
	int64_t taps;
	//! The delta rule's chunk; a compile-time constant (and even), which the forward substitution relies on.
	static constexpr int64_t chunk = 64;
	//! The width of the conv's channels: q, k and v of every head.
	int64_t Channels() const {
		return heads * (2 * key_dim + value_dim);
	}
	//! The width of the layer's input projection, per token: the conv's channels, then z, b and a.
	int64_t Width() const {
		return Channels() + heads * value_dim + 2 * heads;
	}
};

//! Scratch memory for one prompt; reused across calls on one thread.
struct Workspace {
	std::vector<float> conv, decay, beta, row;
	std::vector<float> state, kb, vb, kbd, qd, kd, a, x, u, w, intra, cum, ecum, elast, mask, output;
	void Reserve(const Shape &shape);
};

//! The token mixer and gated norm of a Gated DeltaNet layer (Qwen3.5 linear attention) for one prompt, which follows
//! the tokens that left `initial_state` [H, key_dim, value_dim] (the delta rule's state of each head) and
//! `initial_conv` [taps - 1, C] (the conv's input rows before the prompt); both are zeros at the start of a sequence.
//! `projection` [T, Width()] holds, per token, mixed_qkv [C], z [H * value_dim], b [H] and a [H]:
//!   mixed_qkv -> causal depthwise conv of `taps` [K, C], then SiLU -> q, k, v
//!   q and k L2-normalized per head, q also scaled by key_dim^-0.5
//!   decay = neg_a * softplus(a + dt_bias), beta = sigmoid(b)
//!   o = the gated delta rule over q, k, v, decay and beta
//!   out [T, H * value_dim] = RMSNorm(o) of each head, times norm_weight [value_dim], times SiLU(z)
//! final_state and final_conv get the same for the tokens after the prompt.
//! It computes what transformers' Qwen3_5GatedDeltaNet computes between in_proj and out_proj. The delta rule
//! runs in chunks of `chunk` tokens, in the WY form of transformers' torch_chunk_gated_delta_rule: within a chunk the
//! strictly lower triangular system (I + A) is inverted by forward substitution, and the rest is 8 GEMMs per chunk
//! and head, reading q, k and v in place. A token's output depends only on the tokens up to it, bit for bit.
void Mixer(const Shape &shape, const float *projection, const float *taps, const float *neg_a, const float *dt_bias,
           const float *norm_weight, float eps, const float *initial_state, const float *initial_conv, float *out,
           float *final_state, float *final_conv, Workspace &workspace, const Backend &backend);

} // namespace gated_delta_net
} // namespace decide_onnx
} // namespace duckdb
