#include "decide_onnx/decision_model.hpp"

#include "decide_onnx/model_manifest.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/hash.hpp"

#include <algorithm>
#include <cmath>

namespace duckdb {
namespace decide_onnx {

//! Memoized answers; a few MB for typical prompts.
static constexpr idx_t CACHE_ENTRIES = 100000;
//! The tokens a pack aims for. The GEMMs run in blocks of 32 rows and get faster per row up to about 128; above,
//! attention grows and there is little more to gain.
static constexpr idx_t PACK_TOKENS = 128;

static void Append(std::vector<int64_t> &target, const std::vector<int64_t> &source) {
	target.insert(target.end(), source.begin(), source.end());
}

Prompt EncodePrompt(const BpeTokenizer &tokenizer, const string &state, const string &question,
                    const vector<string> &options) {
	// Text never holds the markers: they come after the tokenizer's vocabulary.
	auto boundary = tokenizer.TokenId("<|endoftext|>");
	Prompt prompt;
	prompt.ids.push_back(boundary);
	prompt.ids.push_back(QUESTION_TOKEN);
	Append(prompt.ids, tokenizer.Encode(question));
	for (auto &option : options) {
		prompt.positions.push_back(static_cast<int64_t>(prompt.ids.size()));
		prompt.ids.push_back(OPTION_TOKEN);
		Append(prompt.ids, tokenizer.Encode(option));
	}
	prompt.ids.push_back(SCHEMA_SEP_TOKEN);
	prompt.state_start = static_cast<int64_t>(prompt.ids.size());
	prompt.ids.push_back(STATE_TOKEN);
	Append(prompt.ids, tokenizer.Encode(state));
	prompt.ids.push_back(boundary);
	if (prompt.ids.size() > CONTEXT_TOKENS) {
		throw InvalidInputException("decide_onnx: the prompt is %llu tokens, more than the %llu-token context of %s. "
		                            "Shorten the input, e.g. left(message, 20000)",
		                            prompt.ids.size(), CONTEXT_TOKENS, MODEL_NAME);
	}
	return prompt;
}

DecisionModel::DecisionModel(shared_ptr<BpeTokenizer> tokenizer_p, shared_ptr<Scorer> scorer_p)
    : tokenizer(std::move(tokenizer_p)), scorer(std::move(scorer_p)) {
}

static uint64_t Fnv1a(const_data_ptr_t data, idx_t size) {
	uint64_t hash = 0xcbf29ce484222325ULL;
	for (idx_t i = 0; i < size; i++) {
		hash = (hash ^ data[i]) * 0x100000001b3ULL;
	}
	return hash;
}

vector<vector<idx_t>> PlanPasses(const vector<idx_t> &lengths, idx_t threads) {
	vector<vector<idx_t>> passes;
	vector<idx_t> packable;
	idx_t total = 0;
	for (idx_t i = 0; i < lengths.size(); i++) {
		if (lengths[i] <= PACK_PROMPT_TOKENS) {
			packable.push_back(i);
			total += lengths[i];
		} else {
			passes.push_back({i});
		}
	}
	// Enough passes to keep every thread busy, as full as PACK_TOKENS allows.
	auto capacity = MinValue<idx_t>(PACK_TOKENS, (total + threads - 1) / MaxValue<idx_t>(threads, 1));
	// First fit decreasing; ties keep the input order, so a chunk always packs the same way.
	std::stable_sort(packable.begin(), packable.end(), [&](idx_t a, idx_t b) { return lengths[a] > lengths[b]; });
	vector<idx_t> filled;
	auto first_pack = passes.size();
	for (auto i : packable) {
		idx_t pack = first_pack;
		while (pack < passes.size() && filled[pack - first_pack] + lengths[i] > capacity) {
			pack++;
		}
		if (pack == passes.size()) {
			passes.emplace_back();
			filled.push_back(0);
		}
		passes[pack].push_back(i);
		filled[pack - first_pack] += lengths[i];
	}
	auto tokens = [&](const vector<idx_t> &pass) {
		idx_t sum = 0;
		for (auto i : pass) {
			sum += lengths[i];
		}
		return sum;
	};
	std::stable_sort(passes.begin(), passes.end(),
	                 [&](const vector<idx_t> &a, const vector<idx_t> &b) { return tokens(a) > tokens(b); });
	return passes;
}

DecisionModel::CacheKey DecisionModel::Key(const Prompt &prompt) {
	// Two independent 64-bit hashes of the whole input stand in for the input itself.
	std::vector<int64_t> key_data(prompt.ids);
	key_data.push_back(-1);
	Append(key_data, prompt.positions);
	key_data.push_back(prompt.state_start);
	auto bytes = reinterpret_cast<const_data_ptr_t>(key_data.data());
	auto size = key_data.size() * sizeof(int64_t);
	return CacheKey {Hash(reinterpret_cast<const char *>(bytes), size), Fnv1a(bytes, size)};
}

vector<std::vector<double>> DecisionModel::Probabilities(const vector<const Prompt *> &prompts) {
	ScorerInput input;
	for (idx_t segment = 0; segment < prompts.size(); segment++) {
		auto &prompt = *prompts[segment];
		D_ASSERT(prompts.size() == 1 || prompt.ids.size() <= PACK_PROMPT_TOKENS);
		auto start = static_cast<int64_t>(input.ids.size());
		auto end = start + static_cast<int64_t>(prompt.ids.size()) - 1;
		Append(input.ids, prompt.ids);
		for (idx_t i = 0; i < prompt.ids.size(); i++) {
			input.position_ids.push_back(static_cast<int64_t>(i));
			input.segment_ids.push_back(static_cast<int64_t>(segment));
		}
		for (auto position : prompt.positions) {
			input.positions.push_back(start + position);
			// EncodePrompt puts [Q] right after the leading <|endoftext|>.
			input.questions.push_back(start + 1);
			input.state_starts.push_back(start + prompt.state_start);
			input.state_ends.push_back(end);
		}
	}
	auto logits = scorer->Logits(input);

	vector<std::vector<double>> results;
	idx_t offset = 0;
	for (auto prompt : prompts) {
		// Softmax in double precision, as the reference does with Python floats.
		auto count = prompt->positions.size();
		double top = -INFINITY;
		for (idx_t k = 0; k < count; k++) {
			top = MaxValue<double>(top, logits[offset + k]);
		}
		double total = 0;
		std::vector<double> result;
		result.reserve(count);
		for (idx_t k = 0; k < count; k++) {
			result.push_back(std::exp(static_cast<double>(logits[offset + k]) - top));
			total += result.back();
		}
		for (auto &p : result) {
			p /= total;
		}
		offset += count;
		Store(Key(*prompt), result);
		results.push_back(std::move(result));
	}
	return results;
}

bool DecisionModel::Lookup(const Prompt &prompt, std::vector<double> &result) {
	auto key = Key(prompt);
	lock_guard<mutex> guard(cache_lock);
	auto entry = cache_index.find(key);
	if (entry == cache_index.end()) {
		return false;
	}
	cache_entries.splice(cache_entries.begin(), cache_entries, entry->second);
	result = entry->second->second;
	return true;
}

void DecisionModel::Store(const CacheKey &key, const std::vector<double> &result) {
	lock_guard<mutex> guard(cache_lock);
	if (cache_index.find(key) != cache_index.end()) {
		return;
	}
	cache_entries.emplace_front(key, result);
	cache_index[key] = cache_entries.begin();
	if (cache_entries.size() > CACHE_ENTRIES) {
		cache_index.erase(cache_entries.back().first);
		cache_entries.pop_back();
	}
}

} // namespace decide_onnx
} // namespace duckdb
