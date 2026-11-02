#pragma once

#include "decide_onnx/cpu_backend.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace decide_onnx {
namespace causal_attention {

//! The sizes of one attention layer over one prompt that follows `past` cached tokens.
struct Shape {
	int64_t tokens;
	int64_t past;
	int64_t heads;
	int64_t kv_heads;
	int64_t head_dim;
	int64_t block = 64;
	//! The prompt's tokens rounded up to whole blocks.
	int64_t Padded() const {
		return (tokens + block - 1) / block * block;
	}
};

//! Scratch memory for one prompt; reused across calls on one thread.
struct Workspace {
	std::vector<float> queries, keys, values, scores, output;
	void Reserve(const Shape &shape);
};

//! Causal grouped-query attention for one prompt that follows `past` cached tokens. q [rows, heads, head_dim] and
//! k, v [rows, kv_heads, head_dim] are row-major, with the prompt in their first `tokens` rows (rows >= tokens);
//! past_k and past_v [>= past, kv_heads, head_dim] hold the cached keys and values. Query head h reads key/value head
//! h / (heads / kv_heads), and the scores are scaled by `scale`.
//! - out [rows, heads, head_dim]: the prompt's rows; the rest are zeros.
//! - present_k, present_v [past + rows, kv_heads, head_dim]: the past keys and values, the prompt's, then zeros.
//! It runs in blocks of `block` queries: one GEMM for the scores of a block against the keys up to its end, a softmax
//! per row over the keys up to it, and one GEMM with the values. Memory grows with past + T, not its square, and
//! every GEMM's shape depends only on `past` and the block's index, so a token's output depends only on the tokens up
//! to it, bit for bit.
void Attention(const Shape &shape, int64_t rows, const float *q, const float *k, const float *v, const float *past_k,
               const float *past_v, float scale, float *out, float *present_k, float *present_v, Workspace &workspace,
               const Backend &backend);

} // namespace causal_attention
} // namespace decide_onnx
} // namespace duckdb
