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

static void GatedDeltaNetMixer(const Ort::Custom::Tensor<float> &projection, const Ort::Custom::Tensor<float> &taps,
                               const Ort::Custom::Tensor<float> &neg_a, const Ort::Custom::Tensor<float> &dt_bias,
                               const Ort::Custom::Tensor<float> &norm_weight, Ort::Custom::Tensor<float> &out) {
	const auto &dims = projection.Shape();
	if (dims.size() != 3 || taps.Shape().size() != 2 || neg_a.Shape().size() != 1 || norm_weight.Shape().size() != 1) {
		ORT_CXX_API_THROW(
		    "GatedDeltaNetMixer: expected projection [B, T, W], taps [K, C], neg_a [H] and norm_weight [D]",
		    ORT_INVALID_ARGUMENT);
	}
	// q, k and v heads of the same size (Qwen3.5: 16 heads of 128).
	const int64_t batch = dims[0], tokens = dims[1], heads = neg_a.Shape()[0], head_dim = norm_weight.Shape()[0];
	gated_delta_net::Shape shape {tokens, heads, head_dim, head_dim, taps.Shape()[0]};
	if (heads <= 0 || head_dim <= 0 || dims[2] != shape.Width() || taps.Shape()[1] != shape.Channels() ||
	    dt_bias.Shape() != neg_a.Shape()) {
		ORT_CXX_API_THROW("GatedDeltaNetMixer: inconsistent input shapes", ORT_INVALID_ARGUMENT);
	}
	const int64_t out_width = heads * head_dim;
	float *output = out.Allocate({batch, tokens, out_width});
	if (tokens == 0) {
		return;
	}
	thread_local gated_delta_net::Workspace workspace;
	const auto &backend = DefaultBackend();
	for (int64_t row = 0; row < batch; row++) {
		gated_delta_net::Mixer(shape, projection.Data() + row * tokens * shape.Width(), taps.Data(), neg_a.Data(),
		                       dt_bias.Data(), norm_weight.Data(), kNormEpsilon, output + row * tokens * out_width,
		                       workspace, backend);
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
                            const Ort::Custom::Tensor<float> &v, Ort::Custom::Tensor<float> &out) {
	const auto &dims = q.Shape();
	if (dims.size() != 4 || k.Shape().size() != 4) {
		ORT_CXX_API_THROW("CausalAttention: expected q [B, T, H, D], k and v [B, T, KV, D]", ORT_INVALID_ARGUMENT);
	}
	const int64_t batch = dims[0], tokens = dims[1], heads = dims[2], dim = dims[3], kv_heads = k.Shape()[2];
	if (kv_heads <= 0 || heads % kv_heads != 0 || k.Shape() != std::vector<int64_t> {batch, tokens, kv_heads, dim} ||
	    v.Shape() != k.Shape()) {
		ORT_CXX_API_THROW("CausalAttention: inconsistent input shapes", ORT_INVALID_ARGUMENT);
	}
	float *output = out.Allocate(dims);
	if (tokens == 0) {
		return;
	}
	causal_attention::Shape shape {tokens, heads, kv_heads, dim};
	const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
	thread_local causal_attention::Workspace workspace;
	const auto &backend = DefaultBackend();
	const int64_t q_size = tokens * heads * dim, kv_size = tokens * kv_heads * dim;
	for (int64_t row = 0; row < batch; row++) {
		causal_attention::Attention(shape, q.Data() + row * q_size, k.Data() + row * kv_size, v.Data() + row * kv_size,
		                            scale, output + row * q_size, workspace, backend);
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
