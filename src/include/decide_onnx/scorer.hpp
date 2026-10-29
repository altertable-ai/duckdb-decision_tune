#pragma once

#include "duckdb/common/common.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace decide_onnx {

//! One forward pass of the vela3-fast encoder plus its decision readout.
class Scorer {
public:
	virtual ~Scorer() = default;
	//! The readout's logit for each of `positions` (the `[O]` before every option) of the sequence `ids`, whose
	//! state starts at `state_start`.
	virtual std::vector<float> Logits(const std::vector<int64_t> &ids, const std::vector<int64_t> &positions,
	                                  int64_t state_start) = 0;
};

//! Runs `model.onnx` (scripts/export/export_onnx.py) with ONNX Runtime (CPU). `Logits` is thread-safe and
//! single-threaded: parallelism comes from DuckDB scoring many prompts at once.
class OrtScorer : public Scorer {
public:
	explicit OrtScorer(const string &onnx_path);
	~OrtScorer() override;

	std::vector<float> Logits(const std::vector<int64_t> &ids, const std::vector<int64_t> &positions,
	                          int64_t state_start) override;

	//! The linked ONNX Runtime version, e.g. "1.29.0".
	static string RuntimeVersion();

private:
	struct Impl;
	unique_ptr<Impl> impl;
};

} // namespace decide_onnx
} // namespace duckdb
