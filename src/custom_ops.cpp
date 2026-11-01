#include "decide_onnx/custom_ops.hpp"

#include "decide_onnx/gated_delta_net.hpp"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>

#include <memory>

// No DuckDB headers here either: scripts/export/custom_ops builds this file into a library for the Python export.

namespace duckdb {
namespace decide_onnx {

static void GatedDeltaNetMixer(const Ort::Custom::Tensor<float> &mixed_qkv, const Ort::Custom::Tensor<float> &a,
                               const Ort::Custom::Tensor<float> &b, const Ort::Custom::Tensor<float> &taps,
                               const Ort::Custom::Tensor<float> &neg_a, const Ort::Custom::Tensor<float> &dt_bias,
                               Ort::Custom::Tensor<float> &out) {
	const auto &dims = mixed_qkv.Shape();
	if (dims.size() != 3 || taps.Shape().size() != 2 || neg_a.Shape().size() != 1) {
		ORT_CXX_API_THROW("GatedDeltaNetMixer: expected mixed_qkv [B, T, C], taps [K, C] and neg_a [H]",
		                  ORT_INVALID_ARGUMENT);
	}
	const int64_t batch = dims[0], tokens = dims[1], channels = dims[2], heads = neg_a.Shape()[0];
	if (heads <= 0 || channels % (3 * heads) != 0 || taps.Shape()[1] != channels ||
	    a.Shape() != std::vector<int64_t> {batch, tokens, heads} || b.Shape() != a.Shape() ||
	    dt_bias.Shape() != neg_a.Shape()) {
		ORT_CXX_API_THROW("GatedDeltaNetMixer: inconsistent input shapes", ORT_INVALID_ARGUMENT);
	}
	// q, k and v heads of the same size (Qwen3.5: 16 heads of 128).
	const int64_t head_dim = channels / (3 * heads);
	gated_delta_net::Shape shape {tokens, heads, head_dim, head_dim, taps.Shape()[0]};
	const int64_t out_width = heads * head_dim;
	float *output = out.Allocate({batch, tokens, out_width});
	if (tokens == 0) {
		return;
	}
	thread_local gated_delta_net::Workspace workspace;
	const auto &backend = gated_delta_net::DefaultBackend();
	for (int64_t row = 0; row < batch; row++) {
		gated_delta_net::Mixer(shape, mixed_qkv.Data() + row * tokens * channels, a.Data() + row * tokens * heads,
		                       b.Data() + row * tokens * heads, taps.Data(), neg_a.Data(), dt_bias.Data(),
		                       output + row * tokens * out_width, workspace, backend);
	}
}

void AddCustomOps(OrtSessionOptions *options) {
	// Leaked on purpose: a domain must outlive every session created with it.
	static auto *op = Ort::Custom::CreateLiteCustomOp("GatedDeltaNetMixer", "CPUExecutionProvider", GatedDeltaNetMixer);
	static auto *domain = [] {
		auto *d = new Ort::CustomOpDomain("decide_onnx");
		d->Add(op);
		return d;
	}();
	Ort::UnownedSessionOptions(options).Add(*domain);
}

} // namespace decide_onnx
} // namespace duckdb
