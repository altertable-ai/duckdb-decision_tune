#include "decide_onnx/gated_delta_net.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {
namespace decide_onnx {
namespace gated_delta_net {

void Workspace::Reserve(const Shape &shape) {
	const size_t tokens = shape.tokens, channels = shape.Channels(), heads = shape.heads, chunk = shape.chunk;
	const size_t key_dim = shape.key_dim, value_dim = shape.value_dim;
	const size_t padded = (tokens + chunk - 1) / chunk * chunk;
	conv.resize(padded * channels), decay.resize(padded * heads), beta.resize(padded * heads);
	row.resize(std::max<size_t>(channels, heads * value_dim));
	output.resize(chunk * value_dim);
	state.resize(key_dim * value_dim);
	kb.resize(chunk * key_dim), kbd.resize(chunk * key_dim), qd.resize(chunk * key_dim), kd.resize(chunk * key_dim);
	w.resize(chunk * key_dim), vb.resize(chunk * value_dim), u.resize(chunk * value_dim);
	a.resize(chunk * chunk), x.resize(chunk * chunk), intra.resize(chunk * chunk), mask.resize(chunk * chunk);
	cum.resize(chunk), ecum.resize(chunk), elast.resize(chunk);
}

//! Sum of squares with independent accumulators, so that it vectorizes without -ffast-math.
static float SumSquares(const float *x, int64_t n) {
	float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	int64_t i = 0;
	for (; i + 8 <= n; i += 8) {
		for (int64_t j = 0; j < 8; j++) {
			acc[j] += x[i + j] * x[i + j];
		}
	}
	float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
	for (; i < n; i++) {
		sum += x[i] * x[i];
	}
	return sum;
}

static float Softplus(float x) {
	// torch.nn.functional.softplus with beta 1 and threshold 20.
	return x > 20.0f ? x : std::log1p(std::exp(x));
}

//! Row `source` of the conv's input: the prompt's projection from 0, the initial conv rows before it.
static const float *ConvInput(const Shape &s, const float *projection, const float *initial_conv, int64_t source) {
	return source < 0 ? initial_conv + (s.taps - 1 + source) * s.Channels() : projection + source * s.Width();
}

//! The causal conv and SiLU of every channel, the L2 norms of the q and k heads, and the gates. The rows past the
//! prompt, up to a whole number of chunks, are zeros: they add nothing to the state and read nothing.
static void Prepare(const Shape &s, const float *projection, const float *initial_conv, const float *taps,
                    const float *neg_a, const float *dt_bias, Workspace &ws, const Backend &be) {
	const int64_t tokens = s.tokens, channels = s.Channels(), num_taps = s.taps, width = s.Width();
	std::vector<const float *> rows(num_taps);
	const float q_scale = 1.0f / std::sqrt(static_cast<float>(s.key_dim));
	float *scratch = ws.row.data();
	for (int64_t t = 0; t < tokens; t++) {
		// The conv reads the current token and the taps - 1 before it, from the initial conv rows before the prompt.
		// One pass over all taps, through restrict pointers so that the compiler keeps the sum in registers.
		float *__restrict y = ws.conv.data() + t * channels;
		for (int64_t j = 0; j < num_taps; j++) {
			rows[j] = ConvInput(s, projection, initial_conv, t - (num_taps - 1) + j);
		}
		if (num_taps == 4) {
			const float *__restrict x0 = rows[0], *__restrict x1 = rows[1], *__restrict x2 = rows[2],
			                        *__restrict x3 = rows[3];
			const float *__restrict w0 = taps, *__restrict w1 = taps + channels, *__restrict w2 = taps + 2 * channels,
			                        *__restrict w3 = taps + 3 * channels;
			for (int64_t c = 0; c < channels; c++) {
				y[c] = (w0[c] * x0[c] + w1[c] * x1[c]) + (w2[c] * x2[c] + w3[c] * x3[c]);
			}
		} else {
			for (int64_t c = 0; c < channels; c++) {
				float sum = 0.0f;
				for (int64_t j = 0; j < num_taps; j++) {
					sum += taps[j * channels + c] * rows[j][c];
				}
				y[c] = sum;
			}
		}
		// SiLU: y / (1 + exp(-y)), with the backend's vector exp.
		for (int64_t c = 0; c < channels; c++) {
			scratch[c] = -y[c];
		}
		be.exp(scratch, scratch, channels);
		for (int64_t c = 0; c < channels; c++) {
			y[c] = y[c] / (1.0f + scratch[c]);
		}
		// transformers' l2norm (eps 1e-6) on each q and k head; q also takes the key_dim^-0.5 of the delta rule.
		for (int64_t part = 0; part < 2; part++) {
			for (int64_t h = 0; h < s.heads; h++) {
				float *x = y + (part * s.heads + h) * s.key_dim;
				const float scale = (part == 0 ? q_scale : 1.0f) / std::sqrt(SumSquares(x, s.key_dim) + 1e-6f);
				for (int64_t d = 0; d < s.key_dim; d++) {
					x[d] *= scale;
				}
			}
		}
		const float *b = projection + t * width + channels + s.heads * s.value_dim, *a = b + s.heads;
		for (int64_t h = 0; h < s.heads; h++) {
			const int64_t i = t * s.heads + h;
			ws.decay[i] = neg_a[h] * Softplus(a[h] + dt_bias[h]);
			ws.beta[i] = 1.0f / (1.0f + std::exp(-b[h]));
		}
	}
	std::fill(ws.conv.begin() + tokens * channels, ws.conv.end(), 0.0f);
	std::fill(ws.decay.begin() + tokens * s.heads, ws.decay.end(), 0.0f);
	std::fill(ws.beta.begin() + tokens * s.heads, ws.beta.end(), 0.0f);
}

//! The gated delta rule of head h over the prepared q, k and v, from its initial state, written to its columns of
//! `out`; its state after the prompt goes to `final_state`. Every chunk is a whole one (the last is padded with the
//! zero rows of Prepare, which leave the state as it is), so that every GEMM has the same shape: a token's output then
//! never depends on the tokens after it, and a prompt in a right-padded batch gets exactly the values it gets alone.
static void DeltaRuleHead(const Shape &s, int64_t h, const float *initial_state, float *out, float *final_state,
                          Workspace &ws, const Backend &be) {
	const int64_t tokens = s.tokens, heads = s.heads, dk = s.key_dim, dv = s.value_dim;
	const int64_t ld = s.Channels(), ld_out = heads * dv;
	const float *q_all = ws.conv.data() + h * dk, *k_all = ws.conv.data() + (heads + h) * dk;
	const float *v_all = ws.conv.data() + 2 * heads * dk + h * dv;
	const int64_t state_size = dk * dv;
	float *state = ws.state.data();
	std::copy(initial_state + h * state_size, initial_state + (h + 1) * state_size, state);
	for (int64_t start = 0; start < tokens; start += s.chunk) {
		const int64_t len = s.chunk, valid = std::min<int64_t>(s.chunk, tokens - start);
		const float *q = q_all + start * ld, *k = k_all + start * ld, *v = v_all + start * ld;
		// cum: the decay accumulated from the start of the chunk, in log space.
		float *cum = ws.cum.data(), *ecum = ws.ecum.data(), *elast = ws.elast.data();
		float running = 0.0f;
		for (int64_t i = 0; i < len; i++) {
			running += ws.decay[(start + i) * heads + h];
			cum[i] = running;
		}
		const float last = cum[len - 1];
		for (int64_t i = 0; i < len; i++) {
			elast[i] = last - cum[i];
		}
		be.exp(cum, ecum, len);
		be.exp(elast, elast, len);
		for (int64_t i = 0; i < len; i++) {
			const float beta = ws.beta[(start + i) * heads + h];
			const float *ki = k + i * ld, *qi = q + i * ld, *vi = v + i * ld;
			float *kb = ws.kb.data() + i * dk, *kbd = ws.kbd.data() + i * dk, *qd = ws.qd.data() + i * dk;
			float *kd = ws.kd.data() + i * dk, *vb = ws.vb.data() + i * dv;
			for (int64_t d = 0; d < dk; d++) {
				kb[d] = ki[d] * beta;
				kbd[d] = kb[d] * ecum[i];
				qd[d] = qi[d] * ecum[i];
				kd[d] = ki[d] * elast[i];
			}
			for (int64_t d = 0; d < dv; d++) {
				vb[d] = vi[d] * beta;
			}
		}
		// A = k_beta k^T for j < i and intra = q k^T for j <= i, each times the decay from token j to token i.
		float *a = ws.a.data(), *intra = ws.intra.data(), *mask = ws.mask.data();
		be.gemm(false, true, len, len, dk, 1.0f, ws.kb.data(), dk, k, ld, 0.0f, a, len);
		be.gemm(false, true, len, len, dk, 1.0f, q, ld, k, ld, 0.0f, intra, len);
		for (int64_t i = 0; i < len; i++) {
			float *m = mask + i * len;
			for (int64_t j = 0; j <= i; j++) {
				m[j] = cum[i] - cum[j];
			}
			be.exp(m, m, i + 1);
			float *ai = a + i * len, *qi = intra + i * len;
			for (int64_t j = 0; j < i; j++) {
				ai[j] *= m[j];
				qi[j] *= m[j];
			}
			qi[i] *= m[i];
			std::fill(ai + i, ai + len, 0.0f);
			std::fill(qi + i + 1, qi + len, 0.0f);
		}
		// x = (I + A)^-1 by forward substitution, two rows at a time: x_i = e_i - sum over m < i of A_im x_m. Each
		// row m is read once for both rows, which accumulate in local arrays. Row m is zero past column
		// m, so the updates may span whole groups of 8 columns, without a remainder loop.
		float *x = ws.x.data();
		static_assert(Shape::chunk % 8 == 0, "the forward substitution works on pairs of rows and groups of 8 columns");
		for (int64_t i = 0; i < len; i += 2) {
			float r0[Shape::chunk] = {}, r1[Shape::chunk] = {};
			for (int64_t m = 0; m < i; m++) {
				const float a0 = a[i * len + m], a1 = a[(i + 1) * len + m];
				const float *xm = x + m * len;
				for (int64_t j = 0; j < ((m + 8) & ~int64_t(7)); j++) {
					r0[j] -= a0 * xm[j];
					r1[j] -= a1 * xm[j];
				}
			}
			r0[i] = 1.0f;
			std::fill(r0 + i + 1, r0 + len, 0.0f);
			std::copy(r0, r0 + len, x + i * len);
			// Row i + 1 also takes row i, the last of its terms.
			const float a10 = a[(i + 1) * len + i];
			for (int64_t j = 0; j <= i; j++) {
				r1[j] -= a10 * r0[j];
			}
			r1[i + 1] = 1.0f;
			std::fill(r1 + i + 2, r1 + len, 0.0f);
			std::copy(r1, r1 + len, x + (i + 1) * len);
		}
		float *u = ws.u.data(), *w = ws.w.data();
		be.gemm(false, false, len, dv, len, 1.0f, x, len, ws.vb.data(), dv, 0.0f, u, dv);  // new values
		be.gemm(false, false, len, dk, len, 1.0f, x, len, ws.kbd.data(), dk, 0.0f, w, dk); // decayed k_beta, solved
		be.gemm(false, false, len, dv, dk, -1.0f, w, dk, state, dv, 1.0f, u, dv);          // v_new = u - w S
		float *o = ws.output.data();
		be.gemm(false, false, len, dv, dk, 1.0f, ws.qd.data(), dk, state, dv, 0.0f, o, dv); // the earlier tokens
		be.gemm(false, false, len, dv, len, 1.0f, intra, len, u, dv, 1.0f, o, dv);          // this chunk
		for (int64_t i = 0; i < valid; i++) {
			std::copy(o + i * dv, o + (i + 1) * dv, out + (start + i) * ld_out + h * dv);
		}
		for (int64_t i = 0; i < dk * dv; i++) {
			state[i] *= ecum[len - 1];
		}
		be.gemm(true, false, dk, dv, len, 1.0f, ws.kd.data(), dk, u, dv, 1.0f, state, dv); // S += k_d^T v_new
	}
	std::copy(state, state + state_size, final_state + h * state_size);
}

//! out = RMSNorm(out) of each head, times the norm's weight, times SiLU(z), token by token.
static void GatedNorm(const Shape &s, const float *projection, const float *weight, float eps, float *out,
                      Workspace &ws, const Backend &be) {
	const int64_t dv = s.value_dim, value_width = s.heads * dv;
	float *gate = ws.row.data();
	for (int64_t t = 0; t < s.tokens; t++) {
		const float *z = projection + t * s.Width() + s.Channels();
		for (int64_t c = 0; c < value_width; c++) {
			gate[c] = -z[c];
		}
		be.exp(gate, gate, value_width);
		for (int64_t h = 0; h < s.heads; h++) {
			float *o = out + t * value_width + h * dv;
			const float *zh = z + h * dv, *eh = gate + h * dv;
			const float scale = 1.0f / std::sqrt(SumSquares(o, dv) / static_cast<float>(dv) + eps);
			for (int64_t d = 0; d < dv; d++) {
				o[d] = weight[d] * (o[d] * scale) * (zh[d] / (1.0f + eh[d]));
			}
		}
	}
}

void Mixer(const Shape &shape, const float *projection, const float *taps, const float *neg_a, const float *dt_bias,
           const float *norm_weight, float eps, const float *initial_state, const float *initial_conv, float *out,
           float *final_state, float *final_conv, Workspace &workspace, const Backend &backend) {
	workspace.Reserve(shape);
	Prepare(shape, projection, initial_conv, taps, neg_a, dt_bias, workspace, backend);
	for (int64_t h = 0; h < shape.heads; h++) {
		DeltaRuleHead(shape, h, initial_state, out, final_state, workspace, backend);
	}
	GatedNorm(shape, projection, norm_weight, eps, out, workspace, backend);
	// The conv's input rows that the next tokens read: the last taps - 1 before the end of the prompt.
	const int64_t channels = shape.Channels();
	for (int64_t j = 0; j < shape.taps - 1; j++) {
		const float *row = ConvInput(shape, projection, initial_conv, shape.tokens - (shape.taps - 1) + j);
		std::copy(row, row + channels, final_conv + j * channels);
	}
}

} // namespace gated_delta_net
} // namespace decide_onnx
} // namespace duckdb
