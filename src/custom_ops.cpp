#include "decide_onnx/custom_ops.hpp"

#include "decide_onnx/causal_attention.hpp"
#include "decide_onnx/elementwise.hpp"
#include "decide_onnx/gated_delta_net.hpp"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>

#include <cmath>
#include <memory>

// No DuckDB headers here either: scripts/export/custom_ops builds this file into a library for the Python export.

namespace duckdb {
namespace decide_onnx {

//! The epsilon of Qwen3.5's RMSNorms; the export checks the model's.
static constexpr float kNormEpsilon = 1e-6f;

//! Checks that `lengths` is [batch], each in [0, max], and returns its values.
static const int64_t *Lengths(const Ort::Custom::Tensor<int64_t> &lengths, int64_t batch, int64_t max,
                              const char *error) {
	if (lengths.Shape() != std::vector<int64_t> {batch}) {
		ORT_CXX_API_THROW(error, ORT_INVALID_ARGUMENT);
	}
	const int64_t *data = lengths.Data();
	for (int64_t row = 0; row < batch; row++) {
		if (data[row] < 0 || data[row] > max) {
			ORT_CXX_API_THROW(error, ORT_INVALID_ARGUMENT);
		}
	}
	return data;
}

static void GatedDeltaNetMixer(const Ort::Custom::Tensor<float> &projection,
                               const Ort::Custom::Tensor<int64_t> &lengths, const Ort::Custom::Tensor<float> &taps,
                               const Ort::Custom::Tensor<float> &neg_a, const Ort::Custom::Tensor<float> &dt_bias,
                               const Ort::Custom::Tensor<float> &norm_weight,
                               const Ort::Custom::Tensor<float> &past_state,
                               const Ort::Custom::Tensor<float> &past_conv, Ort::Custom::Tensor<float> &out,
                               Ort::Custom::Tensor<float> &present_state, Ort::Custom::Tensor<float> &present_conv) {
	const auto &dims = projection.Shape();
	if (dims.size() != 3 || taps.Shape().size() != 2 || neg_a.Shape().size() != 1 || norm_weight.Shape().size() != 1) {
		ORT_CXX_API_THROW(
		    "GatedDeltaNetMixer: expected projection [B, T, W], taps [K, C], neg_a [H] and norm_weight [D]",
		    ORT_INVALID_ARGUMENT);
	}
	// q, k and v heads of the same size (Qwen3.5: 16 heads of 128).
	const int64_t batch = dims[0], tokens = dims[1], heads = neg_a.Shape()[0], head_dim = norm_weight.Shape()[0];
	gated_delta_net::Shape shape {tokens, heads, head_dim, head_dim, taps.Shape()[0]};
	const std::vector<int64_t> state_dims {batch, heads, head_dim, head_dim};
	const std::vector<int64_t> conv_dims {batch, shape.taps - 1, shape.Channels()};
	if (heads <= 0 || head_dim <= 0 || shape.taps <= 0 || dims[2] != shape.Width() ||
	    taps.Shape()[1] != shape.Channels() || dt_bias.Shape() != neg_a.Shape() || past_state.Shape() != state_dims ||
	    past_conv.Shape() != conv_dims) {
		ORT_CXX_API_THROW("GatedDeltaNetMixer: inconsistent input shapes", ORT_INVALID_ARGUMENT);
	}
	const int64_t *length = Lengths(lengths, batch, tokens, "GatedDeltaNetMixer: expected lengths [B] within [0, T]");
	const int64_t out_width = heads * head_dim, state_size = heads * head_dim * head_dim;
	const int64_t conv_size = (shape.taps - 1) * shape.Channels();
	float *output = out.Allocate({batch, tokens, out_width});
	float *final_state = present_state.Allocate(state_dims), *final_conv = present_conv.Allocate(conv_dims);
	thread_local gated_delta_net::Workspace workspace;
	const auto &backend = DefaultBackend();
	for (int64_t row = 0; row < batch; row++) {
		// Each row runs over its own length: its padding is neither computed nor folded into its state.
		shape.tokens = length[row];
		float *row_out = output + row * tokens * out_width;
		gated_delta_net::Mixer(shape, projection.Data() + row * tokens * shape.Width(), taps.Data(), neg_a.Data(),
		                       dt_bias.Data(), norm_weight.Data(), kNormEpsilon, past_state.Data() + row * state_size,
		                       past_conv.Data() + row * conv_size, row_out, final_state + row * state_size,
		                       final_conv + row * conv_size, workspace, backend);
		std::fill(row_out + length[row] * out_width, row_out + tokens * out_width, 0.0f);
	}
}

static void SwiGlu(const Ort::Custom::Tensor<float> &x, Ort::Custom::Tensor<float> &out) {
	auto dims = x.Shape();
	if (dims.empty() || dims.back() % 2 != 0) {
		ORT_CXX_API_THROW("SwiGlu: expected x [..., 2 * I]", ORT_INVALID_ARGUMENT);
	}
	const int64_t width = dims.back() / 2, rows = width == 0 ? 0 : x.NumberOfElement() / (2 * width);
	dims.back() = width;
	float *output = out.Allocate(dims);
	thread_local std::vector<float> scratch;
	elementwise::SwiGlu(x.Data(), rows, width, output, scratch, DefaultBackend());
}

static void AddRmsNorm(const Ort::Custom::Tensor<float> &x, const Ort::Custom::Tensor<float> &delta,
                       const Ort::Custom::Tensor<float> &weight, Ort::Custom::Tensor<float> &sum,
                       Ort::Custom::Tensor<float> &normed) {
	const auto &dims = x.Shape();
	if (dims.empty() || delta.Shape() != dims || weight.Shape() != std::vector<int64_t> {dims.back()}) {
		ORT_CXX_API_THROW("AddRmsNorm: expected x and delta [..., D] and weight [D]", ORT_INVALID_ARGUMENT);
	}
	const int64_t dim = dims.back(), rows = dim == 0 ? 0 : x.NumberOfElement() / dim;
	float *sum_data = sum.Allocate(dims), *normed_data = normed.Allocate(dims);
	elementwise::AddRmsNorm(x.Data(), delta.Data(), weight.Data(), rows, dim, kNormEpsilon, sum_data, normed_data);
}

static void CausalAttention(const Ort::Custom::Tensor<float> &q, const Ort::Custom::Tensor<float> &k,
                            const Ort::Custom::Tensor<float> &v, const Ort::Custom::Tensor<int64_t> &lengths,
                            const Ort::Custom::Tensor<float> &past_key, const Ort::Custom::Tensor<float> &past_value,
                            const Ort::Custom::Tensor<int64_t> &past_lengths, Ort::Custom::Tensor<float> &out,
                            Ort::Custom::Tensor<float> &present_key, Ort::Custom::Tensor<float> &present_value) {
	const auto &dims = q.Shape();
	if (dims.size() != 4 || k.Shape().size() != 4 || past_key.Shape().size() != 4) {
		ORT_CXX_API_THROW("CausalAttention: expected q [B, T, H, D], k and v [B, T, KV, D], past_key and past_value "
		                  "[B, P, KV, D]",
		                  ORT_INVALID_ARGUMENT);
	}
	const int64_t batch = dims[0], tokens = dims[1], heads = dims[2], dim = dims[3], kv_heads = k.Shape()[2];
	const int64_t past = past_key.Shape()[1];
	if (kv_heads <= 0 || heads % kv_heads != 0 || k.Shape() != std::vector<int64_t> {batch, tokens, kv_heads, dim} ||
	    v.Shape() != k.Shape() || past_key.Shape() != std::vector<int64_t> {batch, past, kv_heads, dim} ||
	    past_value.Shape() != past_key.Shape()) {
		ORT_CXX_API_THROW("CausalAttention: inconsistent input shapes", ORT_INVALID_ARGUMENT);
	}
	const int64_t *length = Lengths(lengths, batch, tokens, "CausalAttention: expected lengths [B] within [0, T]");
	const int64_t *past_length =
	    Lengths(past_lengths, batch, past, "CausalAttention: expected past_lengths [B] within [0, P]");
	float *output = out.Allocate(dims);
	const std::vector<int64_t> present_dims {batch, past + tokens, kv_heads, dim};
	float *keys = present_key.Allocate(present_dims), *values = present_value.Allocate(present_dims);
	const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
	thread_local causal_attention::Workspace workspace;
	const auto &backend = DefaultBackend();
	const int64_t q_size = tokens * heads * dim, kv_size = tokens * kv_heads * dim, kv_width = kv_heads * dim;
	const int64_t past_size = past * kv_width, present_size = (past + tokens) * kv_width;
	for (int64_t row = 0; row < batch; row++) {
		causal_attention::Shape shape {length[row], past_length[row], heads, kv_heads, dim};
		float *row_keys = keys + row * present_size, *row_values = values + row * present_size;
		causal_attention::Attention(shape, tokens, q.Data() + row * q_size, k.Data() + row * kv_size,
		                            v.Data() + row * kv_size, past_key.Data() + row * past_size,
		                            past_value.Data() + row * past_size, scale, output + row * q_size, row_keys,
		                            row_values, workspace, backend);
		// A row whose past is shorter than P ends with zeros.
		std::fill(row_keys + (past_length[row] + tokens) * kv_width, row_keys + present_size, 0.0f);
		std::fill(row_values + (past_length[row] + tokens) * kv_width, row_values + present_size, 0.0f);
	}
}

void AddCustomOps(OrtSessionOptions *options) {
	// Leaked on purpose: a domain must outlive every session created with it.
	static auto *mixer =
	    Ort::Custom::CreateLiteCustomOp("GatedDeltaNetMixer", "CPUExecutionProvider", GatedDeltaNetMixer);
	static auto *attention =
	    Ort::Custom::CreateLiteCustomOp("CausalAttention", "CPUExecutionProvider", CausalAttention);
	static auto *swiglu = Ort::Custom::CreateLiteCustomOp("SwiGlu", "CPUExecutionProvider", SwiGlu);
	static auto *add_rms_norm = Ort::Custom::CreateLiteCustomOp("AddRmsNorm", "CPUExecutionProvider", AddRmsNorm);
	static auto *domain = [] {
		auto *d = new Ort::CustomOpDomain("decide_onnx");
		d->Add(mixer);
		d->Add(attention);
		d->Add(swiglu);
		d->Add(add_rms_norm);
		return d;
	}();
	Ort::UnownedSessionOptions(options).Add(*domain);
}

} // namespace decide_onnx
} // namespace duckdb
