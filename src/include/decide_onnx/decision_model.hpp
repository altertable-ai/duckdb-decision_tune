#pragma once

#include "decide_onnx/scorer.hpp"
#include "decide_onnx/tokenizer.hpp"

#include "duckdb/common/list.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/unordered_map.hpp"

#include <utility>
#include <vector>

namespace duckdb {
namespace decide_onnx {

//! The model input for one question:
//! `<|endoftext|> [Q] question ([O] option)+ [SEP_SCHEMA] [SEG_user] state <|endoftext|>`.
struct Prompt {
	std::vector<int64_t> ids;
	//! The position of the `[O]` before each option.
	std::vector<int64_t> positions;
	//! The position of `[SEG_user]`: the state is read from there up to the final `<|endoftext|>`.
	int64_t state_start = 0;
};

//! `DecisionEncoder.encode` of the reference (vela3_encoder.py). Throws InvalidInputException when the prompt
//! exceeds the context (the reference refuses it too, there is no truncation).
Prompt EncodePrompt(const BpeTokenizer &tokenizer, const string &state, const string &question,
                    const vector<string> &options);

//! Groups prompts of `lengths` tokens into forward passes, each a list of indexes into `lengths`. Prompts of at most
//! PACK_PROMPT_TOKENS share passes of about PACK_TOKENS tokens, longer ones run alone; passes are smaller when there
//! would be fewer than `threads` of them. The longest passes come first.
vector<vector<idx_t>> PlanPasses(const vector<idx_t> &lengths, idx_t threads);

//! vela3-fast's decision backbone: the tokenizer, the network and a memo of recent answers. Thread-safe.
class DecisionModel {
public:
	DecisionModel(shared_ptr<BpeTokenizer> tokenizer, shared_ptr<Scorer> scorer);

	const BpeTokenizer &Tokenizer() const {
		return *tokenizer;
	}
	//! The answer to `prompt` if it was computed recently, so a query that evaluates the same decision twice (or runs
	//! again) does not pay for a second forward pass.
	bool Lookup(const Prompt &prompt, std::vector<double> &result);
	//! The softmax over the option logits of each prompt, from one forward pass (see PlanPasses). Remembered for
	//! Lookup.
	vector<std::vector<double>> Probabilities(const vector<const Prompt *> &prompts);

private:
	using CacheKey = std::pair<uint64_t, uint64_t>;
	struct CacheKeyHash {
		size_t operator()(const CacheKey &key) const {
			return static_cast<size_t>(key.first ^ (key.second * 0x9E3779B97F4A7C15ULL));
		}
	};
	using CacheList = list<std::pair<CacheKey, std::vector<double>>>;

	static CacheKey Key(const Prompt &prompt);
	void Store(const CacheKey &key, const std::vector<double> &result);

	shared_ptr<BpeTokenizer> tokenizer;
	shared_ptr<Scorer> scorer;

	mutex cache_lock;
	//! Most recently used first.
	CacheList cache_entries;
	unordered_map<CacheKey, CacheList::iterator, CacheKeyHash> cache_index;
};

} // namespace decide_onnx
} // namespace duckdb
