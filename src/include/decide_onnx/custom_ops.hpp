#pragma once

struct OrtSessionOptions;

namespace duckdb {
namespace decide_onnx {

//! Adds the extension's ONNX Runtime custom ops, domain "decide_onnx", to session options:
//! - GatedDeltaNetMixer(projection [B, T, W], lengths [B] int64, taps [K, C], neg_a [H], dt_bias [H],
//!   norm_weight [D], past_state [B, H, D, D], past_conv [B, K - 1, C])
//!   -> (out [B, T, H * D], present_state [B, H, D, D], present_conv [B, K - 1, C]): the token mixer and gated norm
//!   of a Gated DeltaNet layer (gated_delta_net.hpp) over each row's first `lengths` tokens, after the tokens that
//!   left past_state and past_conv (zeros at the start of a sequence); W = C + H * D + 2 * H and C = 3 * H * D.
//! - CausalAttention(q [B, T, H, D], k [B, T, KV, D], v [B, T, KV, D], lengths [B] int64, past_key [B, P, KV, D],
//!   past_value [B, P, KV, D], past_lengths [B] int64)
//!   -> (out [B, T, H, D], present_key [B, P + T, KV, D], present_value [B, P + T, KV, D]): causal grouped-query
//!   attention scaled by D^-0.5 (causal_attention.hpp) over each row's first `lengths` tokens, after its first
//!   `past_lengths` cached keys and values; each row of present_key and present_value holds those, then the row's
//!   own, then zeros.
//! - SwiGlu(x [..., 2 * I]) -> [..., I]: SiLU of the first half times the second (elementwise.hpp).
//! - AddRmsNorm(x [..., D], delta [..., D], weight [D]) -> (sum, normed), both [..., D]: sum = x + delta and
//!   normed = weight * RMSNorm(sum) (elementwise.hpp).
//! The ops live as long as the process, so sessions may outlive the options.
void AddCustomOps(OrtSessionOptions *options);

} // namespace decide_onnx
} // namespace duckdb
