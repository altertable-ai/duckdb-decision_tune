#include "decide_onnx/tokenizer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "utf8proc.hpp"
#include "utf8proc_wrapper.hpp"
#include "yyjson.hpp"

#include <cstdlib>
#include <cstring>
#include <queue>

namespace duckdb {
namespace decide_onnx {

using namespace duckdb_yyjson; // NOLINT

//===--------------------------------------------------------------------===//
// Character classes of the GPT-2 pre-tokenizer regex
//===--------------------------------------------------------------------===//
static bool IsLetter(int32_t cp) {
	switch (utf8proc_category(cp)) {
	case UTF8PROC_CATEGORY_LU:
	case UTF8PROC_CATEGORY_LL:
	case UTF8PROC_CATEGORY_LT:
	case UTF8PROC_CATEGORY_LM:
	case UTF8PROC_CATEGORY_LO:
		return true;
	default:
		return false;
	}
}

static bool IsNumber(int32_t cp) {
	switch (utf8proc_category(cp)) {
	case UTF8PROC_CATEGORY_ND:
	case UTF8PROC_CATEGORY_NL:
	case UTF8PROC_CATEGORY_NO:
		return true;
	default:
		return false;
	}
}

//! `\s` in Oniguruma's Unicode mode: the White_Space property.
static bool IsSpace(int32_t cp) {
	if (cp < 0x80) {
		return cp == ' ' || (cp >= '\t' && cp <= '\r');
	}
	switch (cp) {
	case 0x85:
	case 0xA0:
	case 0x1680:
	case 0x2028:
	case 0x2029:
	case 0x202F:
	case 0x205F:
	case 0x3000:
		return true;
	default:
		return cp >= 0x2000 && cp <= 0x200A;
	}
}

enum class CharClass : uint8_t { SPACE, LETTER, NUMBER, OTHER };

static CharClass Classify(int32_t cp) {
	if (IsSpace(cp)) {
		return CharClass::SPACE;
	}
	if (IsLetter(cp)) {
		return CharClass::LETTER;
	}
	if (IsNumber(cp)) {
		return CharClass::NUMBER;
	}
	return CharClass::OTHER;
}

//! GPT-2's `bytes_to_unicode`: the code point each byte is shown as in the vocabulary.
static int32_t ByteToCodepoint(uint8_t byte) {
	auto printable = [](int32_t b) {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
	};
	if (printable(byte)) {
		return byte;
	}
	int32_t n = 0;
	for (int32_t b = 0; b < byte; b++) {
		if (!printable(b)) {
			n++;
		}
	}
	return 256 + n;
}

//! The code point at `text[pos]`; throws on invalid UTF-8.
static int32_t NextCodepoint(const char *text, idx_t size, idx_t pos, idx_t &length) {
	int32_t cp;
	auto result = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t *>(text + pos),
	                               NumericCast<utf8proc_ssize_t>(size - pos), &cp);
	if (result <= 0) {
		throw InvalidInputException("decide_onnx: invalid UTF-8 in the text");
	}
	length = NumericCast<idx_t>(result);
	return cp;
}

//! Where the whitespace at the end of `text` starts (`size` if there is none).
static idx_t TrailingSpaceStart(const char *text, idx_t size) {
	idx_t start = 0;
	idx_t pos = 0;
	while (pos < size) {
		idx_t length;
		auto space = IsSpace(NextCodepoint(text, size, pos, length));
		pos += length;
		if (!space) {
			start = pos;
		}
	}
	return start;
}

static string CodepointToUtf8(int32_t cp) {
	char buffer[4];
	int size;
	if (!Utf8Proc::CodepointToUtf8(cp, size, buffer)) {
		throw InternalException("decide_onnx: invalid code point %d", cp);
	}
	return string(buffer, NumericCast<idx_t>(size));
}

//===--------------------------------------------------------------------===//
// tokenizer.json
//===--------------------------------------------------------------------===//
static string JsonString(yyjson_val *val, const string &what, const string &source_name) {
	if (!yyjson_is_str(val)) {
		throw IOException("decide_onnx: %s in '%s' is not a string", what, source_name);
	}
	return string(yyjson_get_str(val), yyjson_get_len(val));
}

static int64_t JsonId(yyjson_val *val, const string &what, const string &source_name) {
	if (!yyjson_is_int(val) || yyjson_get_sint(val) < 0 || yyjson_get_sint(val) > NumericLimits<int32_t>::Maximum()) {
		throw IOException("decide_onnx: %s in '%s' is not a token id", what, source_name);
	}
	return yyjson_get_sint(val);
}

static void ExpectType(yyjson_val *root, const char *key, const char *expected, const string &source_name) {
	auto object = yyjson_obj_get(root, key);
	auto type = object ? yyjson_obj_get(object, "type") : nullptr;
	if (!type || !yyjson_is_str(type) || strcmp(yyjson_get_str(type), expected) != 0) {
		throw IOException("decide_onnx: '%s' is not a vela3-fast tokenizer (%s must be of type %s)", source_name, key,
		                  expected);
	}
}

BpeTokenizer::BpeTokenizer(const string &json, const string &source_name) {
	auto doc = yyjson_read(json.data(), json.size(), YYJSON_READ_NOFLAG);
	if (!doc) {
		throw IOException("decide_onnx: '%s' is not valid JSON", source_name);
	}
	try {
		auto root = yyjson_doc_get_root(doc);
		ExpectType(root, "model", "BPE", source_name);
		auto normalizer = yyjson_obj_get(root, "normalizer");
		nfc = normalizer && !yyjson_is_null(normalizer);
		if (nfc) {
			ExpectType(root, "normalizer", "NFC", source_name);
		}
		ExpectType(root, "pre_tokenizer", "ByteLevel", source_name);
		auto pre_tokenizer = yyjson_obj_get(root, "pre_tokenizer");
		if (yyjson_get_bool(yyjson_obj_get(pre_tokenizer, "add_prefix_space")) ||
		    !yyjson_get_bool(yyjson_obj_get(pre_tokenizer, "use_regex"))) {
			throw IOException("decide_onnx: '%s' uses an unsupported ByteLevel pre-tokenizer configuration",
			                  source_name);
		}
		auto model = yyjson_obj_get(root, "model");
		auto dropout = yyjson_obj_get(model, "dropout");
		if ((dropout && !yyjson_is_null(dropout)) || yyjson_get_bool(yyjson_obj_get(model, "byte_fallback"))) {
			throw IOException("decide_onnx: '%s' uses an unsupported BPE configuration", source_name);
		}

		auto vocab_json = yyjson_obj_get(model, "vocab");
		if (!yyjson_is_obj(vocab_json)) {
			throw IOException("decide_onnx: '%s' has no BPE vocabulary", source_name);
		}
		yyjson_obj_iter vocab_iter;
		yyjson_obj_iter_init(vocab_json, &vocab_iter);
		while (auto key = yyjson_obj_iter_next(&vocab_iter)) {
			vocab[JsonString(key, "a vocabulary token", source_name)] =
			    JsonId(yyjson_obj_iter_get_val(key), "a vocabulary id", source_name);
		}
		for (idx_t byte = 0; byte < 256; byte++) {
			// Bytes that never occur in UTF-8 (0xC0, 0xC1, 0xF5-0xFF) may be missing.
			auto entry = vocab.find(CodepointToUtf8(ByteToCodepoint(static_cast<uint8_t>(byte))));
			byte_ids[byte] = entry == vocab.end() ? -1 : entry->second;
		}

		auto merges_json = yyjson_obj_get(model, "merges");
		if (!yyjson_is_arr(merges_json)) {
			throw IOException("decide_onnx: '%s' has no BPE merges", source_name);
		}
		auto lookup = [&](const string &token) {
			auto entry = vocab.find(token);
			if (entry == vocab.end()) {
				throw IOException("decide_onnx: merge token '%s' is not in the vocabulary of '%s'", token, source_name);
			}
			return entry->second;
		};
		yyjson_arr_iter merge_iter;
		yyjson_arr_iter_init(merges_json, &merge_iter);
		uint32_t rank = 0;
		while (auto merge = yyjson_arr_iter_next(&merge_iter)) {
			string left, right;
			if (yyjson_is_arr(merge) && yyjson_arr_size(merge) == 2) {
				left = JsonString(yyjson_arr_get(merge, 0), "a merge", source_name);
				right = JsonString(yyjson_arr_get(merge, 1), "a merge", source_name);
			} else {
				// The older "left right" spelling
				auto text = JsonString(merge, "a merge", source_name);
				auto space = text.find(' ');
				if (space == string::npos) {
					throw IOException("decide_onnx: invalid merge '%s' in '%s'", text, source_name);
				}
				left = text.substr(0, space);
				right = text.substr(space + 1);
			}
			auto key = (static_cast<uint64_t>(lookup(left)) << 32) | static_cast<uint64_t>(lookup(right));
			// A repeated pair keeps its first (lowest) rank, as in `tokenizers`.
			merges.emplace(key, Merge {rank++, lookup(left + right)});
		}

		auto added_json = yyjson_obj_get(root, "added_tokens");
		if (added_json && yyjson_is_arr(added_json)) {
			yyjson_arr_iter added_iter;
			yyjson_arr_iter_init(added_json, &added_iter);
			while (auto token = yyjson_arr_iter_next(&added_iter)) {
				AddedToken added {JsonString(yyjson_obj_get(token, "content"), "an added token", source_name),
				                  JsonId(yyjson_obj_get(token, "id"), "an added token id", source_name),
				                  yyjson_get_bool(yyjson_obj_get(token, "lstrip")),
				                  yyjson_get_bool(yyjson_obj_get(token, "rstrip"))};
				if (added.content.empty()) {
					continue;
				}
				added_ids[added.content] = added.id;
				if (yyjson_get_bool(yyjson_obj_get(token, "normalized"))) {
					normalized_added_tokens.push_back(std::move(added));
				} else {
					raw_added_tokens.push_back(std::move(added));
				}
			}
		}
	} catch (...) {
		yyjson_doc_free(doc);
		throw;
	}
	yyjson_doc_free(doc);
}

int64_t BpeTokenizer::TokenId(const string &token) const {
	auto added = added_ids.find(token);
	if (added != added_ids.end()) {
		return added->second;
	}
	auto entry = vocab.find(token);
	if (entry == vocab.end()) {
		throw IOException("decide_onnx: the tokenizer has no token '%s'", token);
	}
	return entry->second;
}

//===--------------------------------------------------------------------===//
// Encoding
//===--------------------------------------------------------------------===//
std::vector<int64_t> BpeTokenizer::Encode(const string &text) const {
	std::vector<int64_t> ids;
	SplitOnAddedTokens(text.data(), text.size(), raw_added_tokens, ids,
	                   [&](const char *piece, idx_t size) { EncodeNormalized(piece, size, ids); });
	return ids;
}

void BpeTokenizer::EncodeNormalized(const char *text, idx_t size, std::vector<int64_t> &ids) const {
	if (!nfc) {
		SplitOnAddedTokens(text, size, normalized_added_tokens, ids,
		                   [&](const char *piece, idx_t piece_size) { EncodePreTokenized(piece, piece_size, ids); });
		return;
	}
	auto normalized = Utf8Proc::Normalize(text, size);
	if (!normalized) {
		throw InvalidInputException("decide_onnx: could not normalize the text to Unicode NFC");
	}
	string owned(normalized);
	free(normalized);
	SplitOnAddedTokens(owned.data(), owned.size(), normalized_added_tokens, ids,
	                   [&](const char *piece, idx_t piece_size) { EncodePreTokenized(piece, piece_size, ids); });
}

template <class ON_TEXT>
void BpeTokenizer::SplitOnAddedTokens(const char *text, idx_t size, const std::vector<AddedToken> &tokens,
                                      std::vector<int64_t> &ids, ON_TEXT &&on_text) const {
	// Leftmost-longest, like the Aho-Corasick matcher of `tokenizers`. The added tokens are few.
	idx_t segment_start = 0;
	idx_t pos = 0;
	while (pos < size) {
		const AddedToken *best = nullptr;
		for (auto &token : tokens) {
			auto length = token.content.size();
			if (length <= size - pos && (!best || length > best->content.size()) &&
			    memcmp(text + pos, token.content.data(), length) == 0) {
				best = &token;
			}
		}
		if (!best) {
			pos++;
			continue;
		}
		// lstrip / rstrip: the token also takes the whitespace before / after it.
		auto text_end =
		    best->lstrip ? segment_start + TrailingSpaceStart(text + segment_start, pos - segment_start) : pos;
		if (text_end > segment_start) {
			on_text(text + segment_start, text_end - segment_start);
		}
		ids.push_back(best->id);
		pos += best->content.size();
		while (best->rstrip && pos < size) {
			idx_t length;
			if (!IsSpace(NextCodepoint(text, size, pos, length))) {
				break;
			}
			pos += length;
		}
		segment_start = pos;
	}
	if (size > segment_start) {
		on_text(text + segment_start, size - segment_start);
	}
}

//! Splits like the GPT-2 regex
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
void BpeTokenizer::EncodePreTokenized(const char *text, idx_t size, std::vector<int64_t> &ids) const {
	std::vector<int32_t> cps;
	std::vector<CharClass> classes;
	std::vector<idx_t> offsets;
	idx_t pos = 0;
	while (pos < size) {
		idx_t length;
		auto cp = NextCodepoint(text, size, pos, length);
		cps.push_back(cp);
		classes.push_back(Classify(cp));
		offsets.push_back(pos);
		pos += length;
	}
	offsets.push_back(size);
	auto n = cps.size();
	auto emit = [&](idx_t from, idx_t to) {
		EncodeWord(text + offsets[from], offsets[to] - offsets[from], ids);
	};
	auto run_end = [&](idx_t from, CharClass cls) {
		while (from < n && classes[from] == cls) {
			from++;
		}
		return from;
	};

	idx_t i = 0;
	while (i < n) {
		auto c = cps[i];
		if (c == '\'' && i + 1 < n) {
			auto c1 = cps[i + 1];
			if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
				emit(i, i + 2);
				i += 2;
				continue;
			}
			if (i + 2 < n && ((c1 == 'r' && cps[i + 2] == 'e') || (c1 == 'v' && cps[i + 2] == 'e') ||
			                  (c1 == 'l' && cps[i + 2] == 'l'))) {
				emit(i, i + 3);
				i += 3;
				continue;
			}
		}
		// ` ?\p{L}+`, ` ?\p{N}+` and ` ?[^\s\p{L}\p{N}]+`
		auto body = (c == ' ' && i + 1 < n && classes[i + 1] != CharClass::SPACE) ? i + 1 : i;
		if (classes[body] != CharClass::SPACE) {
			auto end = run_end(body, classes[body]);
			emit(i, end);
			i = end;
			continue;
		}
		// `\s+(?!\S)`, else `\s+`
		auto end = run_end(i, CharClass::SPACE);
		if (end < n && end - i >= 2) {
			end--;
		}
		emit(i, end);
		i = end;
	}
}

void BpeTokenizer::EncodeWord(const char *word, idx_t size, std::vector<int64_t> &ids) const {
	struct Symbol {
		int64_t id;
		int64_t prev;
		int64_t next;
		bool alive;
	};
	struct Candidate {
		uint32_t rank;
		idx_t pos;
		int64_t id;
		bool operator>(const Candidate &other) const {
			return rank != other.rank ? rank > other.rank : pos > other.pos;
		}
	};
	std::vector<Symbol> symbols(size);
	for (idx_t i = 0; i < size; i++) {
		auto id = byte_ids[static_cast<uint8_t>(word[i])];
		if (id < 0) {
			throw InvalidInputException("decide_onnx: byte 0x%02x is not in the tokenizer vocabulary",
			                            static_cast<uint8_t>(word[i]));
		}
		symbols[i] = Symbol {id, static_cast<int64_t>(i) - 1, i + 1 < size ? static_cast<int64_t>(i + 1) : -1, true};
	}
	// Lowest rank first, leftmost on ties, as `tokenizers` does.
	std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
	auto push_pair = [&](int64_t left) {
		if (left < 0 || symbols[left].next < 0) {
			return;
		}
		auto key =
		    (static_cast<uint64_t>(symbols[left].id) << 32) | static_cast<uint64_t>(symbols[symbols[left].next].id);
		auto merge = merges.find(key);
		if (merge != merges.end()) {
			queue.push(Candidate {merge->second.rank, static_cast<idx_t>(left), merge->second.id});
		}
	};
	for (idx_t i = 0; i + 1 < size; i++) {
		push_pair(static_cast<int64_t>(i));
	}
	while (!queue.empty()) {
		auto top = queue.top();
		queue.pop();
		auto &left = symbols[top.pos];
		if (!left.alive || left.next < 0) {
			continue;
		}
		auto right_pos = left.next;
		auto key = (static_cast<uint64_t>(left.id) << 32) | static_cast<uint64_t>(symbols[right_pos].id);
		auto merge = merges.find(key);
		if (merge == merges.end() || merge->second.rank != top.rank || merge->second.id != top.id) {
			continue; // stale: one side was merged since
		}
		left.id = top.id;
		auto &right = symbols[right_pos];
		right.alive = false;
		left.next = right.next;
		if (right.next >= 0) {
			symbols[right.next].prev = static_cast<int64_t>(top.pos);
		}
		push_pair(left.prev);
		push_pair(static_cast<int64_t>(top.pos));
	}
	for (int64_t i = size > 0 ? 0 : -1; i >= 0; i = symbols[i].next) {
		ids.push_back(symbols[i].id);
	}
}

} // namespace decide_onnx
} // namespace duckdb
