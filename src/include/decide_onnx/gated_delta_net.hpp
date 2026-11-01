#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// This header and gated_delta_net.cpp depend on neither DuckDB nor ONNX Runtime's headers, so that
// scripts/export/custom_ops builds them into the library the export checks its graphs with.

namespace duckdb {
namespace decide_onnx {
namespace gated_delta_net {

//! C = alpha * op(A) * op(B) + beta * C, row-major.
using Sgemm = void (*)(bool trans_a, bool trans_b, size_t m, size_t n, size_t k, float alpha, const float *a,
                       size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc);
//! y[i] = exp(x[i]) for n values; x and y may be the same array.
using VecExp = void (*)(const float *x, float *y, size_t n);

//! The SGEMM and vector exp the kernel runs on: Accelerate on Apple platforms, ONNX Runtime's MLAS elsewhere.
struct Backend {
	Sgemm gemm;
	VecExp exp;
};

//! The default backend of this build.
const Backend &DefaultBackend();

//! The sizes of one Gated DeltaNet layer over one prompt.
struct Shape {
	int64_t tokens;
	int64_t heads;
	int64_t key_dim;
	int64_t value_dim;
	int64_t taps;
	int64_t chunk = 64;
	//! The width of the input projection the mixer reads: q, k and v of every head.
	int64_t Channels() const {
		return heads * (2 * key_dim + value_dim);
	}
};

//! Scratch memory for one prompt; reused across calls on one thread.
struct Workspace {
	std::vector<float> conv, decay, beta, row;
	std::vector<float> state, kb, vb, kbd, qd, kd, a, x, u, w, intra, cum, ecum, elast, mask, output;
	void Reserve(const Shape &shape);
};

//! The token mixer of a Gated DeltaNet layer (Qwen3.5 linear attention) for one prompt, from a zero state:
//!   mixed_qkv [T, C] -> causal depthwise conv of `taps` [K, C], then SiLU -> q, k, v
//!   q and k L2-normalized per head, q also scaled by key_dim^-0.5
//!   decay = neg_a * softplus(a + dt_bias), beta = sigmoid(b)                      a, b: [T, H]
//!   out [T, H * value_dim] = the gated delta rule over q, k, v, decay and beta
//! It computes what transformers' Qwen3_5GatedDeltaNet computes between in_proj and its gated norm. The delta rule
//! runs in chunks of `chunk` tokens, in the WY form of transformers' torch_chunk_gated_delta_rule: within a chunk the
//! strictly lower triangular system (I + A) is inverted by forward substitution, and the rest is 8 GEMMs per chunk
//! and head, reading q, k and v in place. A token's output depends only on the tokens up to it, bit for bit.
void Mixer(const Shape &shape, const float *mixed_qkv, const float *a, const float *b, const float *taps,
           const float *neg_a, const float *dt_bias, float *out, Workspace &workspace, const Backend &backend);

} // namespace gated_delta_net
} // namespace decide_onnx
} // namespace duckdb
