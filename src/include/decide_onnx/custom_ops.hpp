#pragma once

struct OrtSessionOptions;

namespace duckdb {
namespace decide_onnx {

//! Adds the extension's ONNX Runtime custom ops, domain "decide_onnx", to session options:
//! - GatedDeltaNetMixer(projection [B, T, W], taps [K, C], neg_a [H], dt_bias [H], norm_weight [D])
//!   -> [B, T, H * D]: the token mixer and gated norm of a Gated DeltaNet layer (gated_delta_net.hpp), for each row of
//!   the batch; W = C + H * D + 2 * H and C = 3 * H * D.
//! - CausalAttention(q [B, T, H, D], k [B, T, KV, D], v [B, T, KV, D]) -> [B, T, H, D]: causal grouped-query
//!   attention scaled by D^-0.5 (causal_attention.hpp), for each row of the batch.
//! - SwiGlu(x [..., 2 * I]) -> [..., I]: SiLU of the first half times the second (elementwise.hpp).
//! - AddRmsNorm(x [..., D], delta [..., D], weight [D]) -> (sum, normed), both [..., D]: sum = x + delta and
//!   normed = weight * RMSNorm(sum) (elementwise.hpp).
//! The ops live as long as the process, so sessions may outlive the options.
void AddCustomOps(OrtSessionOptions *options);

} // namespace decide_onnx
} // namespace duckdb
