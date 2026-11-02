#include "decide_onnx/scorer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#ifdef _WIN32
#include "duckdb/common/windows_util.hpp"
#endif

#include <onnxruntime_cxx_api.h>

namespace duckdb {
namespace decide_onnx {

static Ort::Env &SharedEnv() {
	// One environment per process; it owns ONNX Runtime's logging and global thread state. Never destroyed, so it
	// outlives every session (sessions live in the never-destroyed ModelStore).
	static auto *env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "decide_onnx");
	return *env;
}

struct OrtScorer::Impl {
	unique_ptr<Ort::Session> session;
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

OrtScorer::OrtScorer(const string &onnx_path) : impl(make_uniq<Impl>()) {
	try {
		Ort::SessionOptions options;
		options.SetIntraOpNumThreads(1);
		options.SetInterOpNumThreads(1);
		options.SetExecutionMode(ORT_SEQUENTIAL);
		options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		// DuckDB already keeps its own threads busy; spinning ONNX Runtime workers would only steal CPU.
		options.AddConfigEntry("session.intra_op.allow_spinning", "0");
		// The weights ship as int8 DequantizeLinear initializers. Without this, ONNX Runtime fuses them into
		// MatMulNBits, several times slower on CPU; with it, constant folding turns them back into fp32 at load.
		options.AddConfigEntry("session.disable_quant_qdq", "1");
#ifdef _WIN32
		auto wide_path = WindowsUtil::UTF8ToUnicode(onnx_path.c_str());
		impl->session = make_uniq<Ort::Session>(SharedEnv(), wide_path.c_str(), options);
#else
		impl->session = make_uniq<Ort::Session>(SharedEnv(), onnx_path.c_str(), options);
#endif
	} catch (const Ort::Exception &ex) {
		throw IOException("decide_onnx: could not load model '%s': %s", onnx_path, ex.what());
	}
}

OrtScorer::~OrtScorer() = default;

std::vector<float> OrtScorer::Logits(const ScorerInput &input) {
	int64_t sequence_shape[] = {1, NumericCast<int64_t>(input.ids.size())};
	int64_t options_shape[] = {NumericCast<int64_t>(input.positions.size())};
	auto &mem = impl->memory_info;
	auto tensor = [&](const std::vector<int64_t> &data, int64_t *shape, size_t rank) {
		return Ort::Value::CreateTensor<int64_t>(mem, const_cast<int64_t *>(data.data()), data.size(), shape, rank);
	};
	try {
		Ort::Value values[] = {
		    tensor(input.ids, sequence_shape, 2),         tensor(input.position_ids, sequence_shape, 2),
		    tensor(input.segment_ids, sequence_shape, 2), tensor(input.positions, options_shape, 1),
		    tensor(input.questions, options_shape, 1),    tensor(input.state_starts, options_shape, 1),
		    tensor(input.state_ends, options_shape, 1)};
		const char *input_names[] = {"input_ids", "position_ids", "segment_ids", "positions",
		                             "questions", "state_starts", "state_ends"};
		const char *output_names[] = {"logits"};
		auto outputs = impl->session->Run(Ort::RunOptions {nullptr}, input_names, values, 7, output_names, 1);
		auto info = outputs[0].GetTensorTypeAndShapeInfo();
		if (info.GetElementCount() != input.positions.size()) {
			throw InternalException("decide_onnx: model returned %llu logits, expected %llu", info.GetElementCount(),
			                        input.positions.size());
		}
		auto data = outputs[0].GetTensorData<float>();
		return std::vector<float>(data, data + input.positions.size());
	} catch (const Ort::Exception &ex) {
		throw InvalidInputException("decide_onnx: model inference failed: %s", ex.what());
	}
}

string OrtScorer::RuntimeVersion() {
	return OrtGetApiBase()->GetVersionString();
}

} // namespace decide_onnx
} // namespace duckdb
