#include "decide_onnx/causal_attention.hpp"

#include <algorithm>

namespace duckdb {
namespace decide_onnx {
namespace causal_attention {

void Workspace::Reserve(const Shape &shape) {
	const size_t padded = shape.Padded(), keys_rows = shape.past + padded, dim = shape.head_dim;
	queries.resize(padded * dim), keys.resize(keys_rows * dim), values.resize(keys_rows * dim);
	scores.resize(shape.block * keys_rows), output.resize(shape.block * dim);
}

//! Copies head `head` of the first `count` rows of [rows, heads, dim] into [count, dim], then zeros up to `to`'s end.
static void GatherHead(const float *x, int64_t count, int64_t heads, int64_t head, int64_t dim,
                       std::vector<float> &to) {
	for (int64_t t = 0; t < count; t++) {
		const float *row = x + (t * heads + head) * dim;
		std::copy(row, row + dim, to.data() + t * dim);
	}
	std::fill(to.begin() + count * dim, to.end(), 0.0f);
}

void Attention(const Shape &s, int64_t rows, const float *q, const float *k, const float *v, const float *past_k,
               const float *past_v, float scale, float *out, float *present_k, float *present_v, Workspace &ws,
               const Backend &be) {
	ws.Reserve(s);
	const int64_t tokens = s.tokens, past = s.past, dim = s.head_dim, block = s.block, group = s.heads / s.kv_heads;
	const int64_t padded = s.Padded(), kv_width = s.kv_heads * dim;
	// present = the past keys and values, the prompt's, then zeros.
	auto concatenate = [&](const float *cached, const float *prompt, float *present) {
		std::copy(cached, cached + past * kv_width, present);
		std::copy(prompt, prompt + tokens * kv_width, present + past * kv_width);
		std::fill(present + (past + tokens) * kv_width, present + (past + rows) * kv_width, 0.0f);
	};
	concatenate(past_k, k, present_k);
	concatenate(past_v, v, present_v);
	std::fill(out, out + rows * s.heads * dim, 0.0f);
	for (int64_t kv = 0; kv < s.kv_heads; kv++) {
		GatherHead(present_k, past + tokens, s.kv_heads, kv, dim, ws.keys);
		GatherHead(present_v, past + tokens, s.kv_heads, kv, dim, ws.values);
		for (int64_t h = kv * group; h < (kv + 1) * group; h++) {
			GatherHead(q, tokens, s.heads, h, dim, ws.queries);
			for (int64_t start = 0; start < padded; start += block) {
				// The block's queries against every key up to its last query; row i sees the keys up to past + start +
				// i.
				const int64_t end = past + start + block, valid = std::min(block, tokens - start);
				float *scores = ws.scores.data();
				be.gemm(false, true, block, end, dim, scale, ws.queries.data() + start * dim, dim, ws.keys.data(), dim,
				        0.0f, scores, end);
				for (int64_t i = 0; i < block; i++) {
					float *row = scores + i * end;
					const int64_t seen = past + start + i + 1;
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
