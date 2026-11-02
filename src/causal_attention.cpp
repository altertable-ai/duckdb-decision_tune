#include "decide_onnx/causal_attention.hpp"

#include <algorithm>

namespace duckdb {
namespace decide_onnx {
namespace causal_attention {

void Workspace::Reserve(const Shape &shape) {
	const size_t padded = (shape.tokens + shape.block - 1) / shape.block * shape.block, dim = shape.head_dim;
	queries.resize(padded * dim), keys.resize(padded * dim), values.resize(padded * dim);
	scores.resize(shape.block * padded), output.resize(shape.block * dim);
}

//! Copies head `head` of [T, heads, dim] into [padded, dim]; the rows past T are zeros.
static void GatherHead(const float *x, int64_t tokens, int64_t heads, int64_t head, int64_t dim,
                       std::vector<float> &to) {
	for (int64_t t = 0; t < tokens; t++) {
		const float *row = x + (t * heads + head) * dim;
		std::copy(row, row + dim, to.data() + t * dim);
	}
	std::fill(to.begin() + tokens * dim, to.end(), 0.0f);
}

void Attention(const Shape &s, const float *q, const float *k, const float *v, float scale, float *out, Workspace &ws,
               const Backend &be) {
	ws.Reserve(s);
	const int64_t tokens = s.tokens, dim = s.head_dim, block = s.block, group = s.heads / s.kv_heads;
	const int64_t padded = (tokens + block - 1) / block * block;
	for (int64_t kv = 0; kv < s.kv_heads; kv++) {
		GatherHead(k, tokens, s.kv_heads, kv, dim, ws.keys);
		GatherHead(v, tokens, s.kv_heads, kv, dim, ws.values);
		for (int64_t h = kv * group; h < (kv + 1) * group; h++) {
			GatherHead(q, tokens, s.heads, h, dim, ws.queries);
			for (int64_t start = 0; start < padded; start += block) {
				// The block's queries against every key up to its last query; row i sees the keys up to start + i.
				const int64_t end = start + block, valid = std::min(block, tokens - start);
				float *scores = ws.scores.data();
				be.gemm(false, true, block, end, dim, scale, ws.queries.data() + start * dim, dim, ws.keys.data(), dim,
				        0.0f, scores, end);
				for (int64_t i = 0; i < block; i++) {
					float *row = scores + i * end;
					const int64_t seen = start + i + 1;
					const float max = *std::max_element(row, row + seen);
					for (int64_t j = 0; j < seen; j++) {
						row[j] -= max;
					}
					be.exp(row, row, seen);
					float sum = 0.0f;
					for (int64_t j = 0; j < seen; j++) {
						sum += row[j];
					}
					const float inverse = 1.0f / sum;
					for (int64_t j = 0; j < seen; j++) {
						row[j] *= inverse;
					}
					std::fill(row + seen, row + end, 0.0f);
				}
				float *o = ws.output.data();
				be.gemm(false, false, block, dim, end, 1.0f, scores, end, ws.values.data(), dim, 0.0f, o, dim);
				for (int64_t i = 0; i < valid; i++) {
					std::copy(o + i * dim, o + (i + 1) * dim, out + ((start + i) * s.heads + h) * dim);
				}
			}
		}
	}
}

} // namespace causal_attention
} // namespace decide_onnx
} // namespace duckdb
