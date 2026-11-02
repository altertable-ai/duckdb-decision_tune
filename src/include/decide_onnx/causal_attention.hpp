#pragma once

#include "decide_onnx/cpu_backend.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace decide_onnx {
namespace causal_attention {

//! The sizes of one attention layer over one prompt.
struct Shape {
	int64_t tokens;
	int64_t heads;
	int64_t kv_heads;
	int64_t head_dim;
	int64_t block = 64;
};

//! Scratch memory for one prompt; reused across calls on one thread.
struct Workspace {
	std::vector<float> queries, keys, values, scores, output;
	void Reserve(const Shape &shape);
};

//! Causal grouped-query attention for one prompt, from row-major q [T, heads, head_dim] and k, v [T, kv_heads,
//! head_dim] to out [T, heads, head_dim]; query head h reads key/value head h / (heads / kv_heads), and the scores
//! are scaled by `scale`. It runs in blocks of `block` queries: one GEMM for the scores of a block against the keys
//! up to its end, a softmax per row over the keys up to it, and one GEMM with the values. Memory grows with T, not
//! T^2, and every GEMM's shape depends only on the block's index, so a token's output depends only on the tokens up
//! to it, bit for bit.
void Attention(const Shape &shape, const float *q, const float *k, const float *v, float scale, float *out,
               Workspace &workspace, const Backend &backend);

} // namespace causal_attention
} // namespace decide_onnx
} // namespace duckdb
