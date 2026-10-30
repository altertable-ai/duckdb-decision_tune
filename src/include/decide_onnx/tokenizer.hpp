#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace decide_onnx {

//! The byte-level BPE tokenizer of vela3-fast (UltraBERT's), read from its Hugging Face `tokenizer.json`.
//! A C++ port of the subset of `tokenizers` such files use: added tokens (special ones matched on the raw text,
//! the others after normalization; lstrip and rstrip), no or NFC normalization, the GPT-2 ByteLevel pre-tokenizer
//! and BPE. Thread-safe.
class BpeTokenizer {
public:
	//! Parses the contents of a `tokenizer.json`. Throws IOException if it is not the expected kind of tokenizer.
	BpeTokenizer(const string &json, const string &source_name);

	//! `tokenizer.encode(text, add_special_tokens=False).ids`. `text` must be valid UTF-8.
	std::vector<int64_t> Encode(const string &text) const;
	//! The id of a vocabulary or added token; throws IOException if it is missing.
	int64_t TokenId(const string &token) const;

private:
	struct AddedToken {
		string content;
		int64_t id;
		bool lstrip;
		bool rstrip;
	};
	struct Merge {
		uint32_t rank;
		int64_t id;
	};

	//! Appends the ids of `text` (no added tokens in it) after normalization and normalized added-token splits.
	void EncodeNormalized(const char *text, idx_t size, std::vector<int64_t> &ids) const;
	//! Splits `text` on `tokens` (leftmost-longest), calling `on_text` for the text between matches.
	template <class ON_TEXT>
	void SplitOnAddedTokens(const char *text, idx_t size, const std::vector<AddedToken> &tokens,
	                        std::vector<int64_t> &ids, ON_TEXT &&on_text) const;
	//! Pre-tokenizes `text` with the GPT-2 regex and runs BPE on every piece.
	void EncodePreTokenized(const char *text, idx_t size, std::vector<int64_t> &ids) const;
	void EncodeWord(const char *word, idx_t size, std::vector<int64_t> &ids) const;

	unordered_map<string, int64_t> vocab;
	//! Rank and result of merging the tokens `(left << 32) | right`.
	unordered_map<uint64_t, Merge> merges;
	//! The vocabulary id of each byte, after the GPT-2 byte-to-unicode mapping; -1 if it has none.
	int64_t byte_ids[256];
	//! Whether the text is NFC-normalized (`normalizer` in tokenizer.json, else none).
	bool nfc = false;
	//! Matched on the raw text (`normalized: false` in tokenizer.json).
	std::vector<AddedToken> raw_added_tokens;
	//! Matched on the NFC-normalized text (`normalized: true`).
	std::vector<AddedToken> normalized_added_tokens;
	//! Every added token, for TokenId.
	unordered_map<string, int64_t> added_ids;
};

} // namespace decide_onnx
} // namespace duckdb
