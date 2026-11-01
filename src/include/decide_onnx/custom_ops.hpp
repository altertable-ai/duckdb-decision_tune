#pragma once

struct OrtSessionOptions;

namespace duckdb {
namespace decide_onnx {

//! Adds the extension's ONNX Runtime custom ops, domain "decide_onnx", to session options:
//! - GatedDeltaNetMixer(mixed_qkv [B, T, C], a [B, T, H], b [B, T, H], taps [K, C], neg_a [H], dt_bias [H])
//!   -> [B, T, C / 3]: the token mixer of a Gated DeltaNet layer (gated_delta_net.hpp), for each row of the batch.
//! The ops live as long as the process, so sessions may outlive the options.
void AddCustomOps(OrtSessionOptions *options);

} // namespace decide_onnx
} // namespace duckdb
