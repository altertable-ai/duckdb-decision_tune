#pragma once

#include "duckdb/common/common.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace decide_onnx {

//! The input of one forward pass: one prompt, or a pack of short prompts laid end to end (see
//! scripts/export/export_onnx.py). Every position is an index into `ids`.
struct ScorerInput {
	std::vector<int64_t> ids;
	//! Each token's position within its prompt.
	std::vector<int64_t> position_ids;
	//! Each token's prompt: 0, 1, ... in order.
	std::vector<int64_t> segment_ids;
	//! Per option, over all prompts: its `[O]`, and the `[Q]`, `[SEG_user]` and final `<|endoftext|>` of its prompt.
	std::vector<int64_t> positions;
	std::vector<int64_t> questions;
	std::vector<int64_t> state_starts;
	std::vector<int64_t> state_ends;
};

//! One forward pass of the vela3-fast encoder plus its decision readout.
class Scorer {
public:
	virtual ~Scorer() = default;
	//! The readout's logit for each option of `input`.
	virtual std::vector<float> Logits(const ScorerInput &input) = 0;
};

//! Runs `model.onnx` (scripts/export/export_onnx.py) with ONNX Runtime (CPU). `Logits` is thread-safe and
//! single-threaded: parallelism comes from DuckDB scoring many prompts at once.
class OrtScorer : public Scorer {
public:
	explicit OrtScorer(const string &onnx_path);
	~OrtScorer() override;

	std::vector<float> Logits(const ScorerInput &input) override;

	//! The linked ONNX Runtime version, e.g. "1.29.0".
	static string RuntimeVersion();

private:
	struct Impl;
	unique_ptr<Impl> impl;
};

} // namespace decide_onnx
} // namespace duckdb
