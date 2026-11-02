#include "decide_onnx/state_text.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "yyjson.hpp"

#include <cmath>
#include <cstdlib>

namespace duckdb {
namespace decide_onnx {

using namespace duckdb_yyjson; // NOLINT

//===--------------------------------------------------------------------===//
// Compact JSON, as json.dumps(value, ensure_ascii=False, separators=(",", ":"))
//===--------------------------------------------------------------------===//
static void AppendJsonString(string &out, const string &text) {
	out += '"';
	for (auto c : text) {
		auto byte = static_cast<unsigned char>(c);
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		case '\b':
			out += "\\b";
			break;
		case '\f':
			out += "\\f";
			break;
		default:
			if (byte < 0x20) {
				out += StringUtil::Format("\\u%04x", byte);
			} else {
				out += c;
			}
		}
	}
	out += '"';
}

static bool IsTextType(const LogicalType &type) {
	return type.id() == LogicalTypeId::VARCHAR || type.id() == LogicalTypeId::ENUM;
}

static bool IsNumericType(const LogicalType &type) {
	return type.IsNumeric() || type.id() == LogicalTypeId::BOOLEAN;
}

static void AppendJson(string &out, const Value &value);

static void AppendJsonNumber(string &out, const Value &value) {
	auto &type = value.type();
	if (type.id() == LogicalTypeId::BOOLEAN) {
		out += BooleanValue::Get(value) ? "true" : "false";
		return;
	}
	if (type.id() == LogicalTypeId::DOUBLE || type.id() == LogicalTypeId::FLOAT) {
		auto number = value.GetValue<double>();
		if (std::isnan(number)) {
			out += "NaN";
			return;
		}
		if (std::isinf(number)) {
			out += number > 0 ? "Infinity" : "-Infinity";
			return;
		}
	}
	out += value.ToString();
}

static void AppendJson(string &out, const Value &value) {
	if (value.IsNull()) {
		out += "null";
		return;
	}
	auto &type = value.type();
	if (IsNumericType(type)) {
		AppendJsonNumber(out, value);
		return;
	}
	switch (type.id()) {
	case LogicalTypeId::LIST:
	case LogicalTypeId::ARRAY: {
		auto &children =
		    type.id() == LogicalTypeId::LIST ? ListValue::GetChildren(value) : ArrayValue::GetChildren(value);
		out += '[';
		for (idx_t i = 0; i < children.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			AppendJson(out, children[i]);
		}
		out += ']';
		return;
	}
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::TUPLE: {
		auto &children = StructValue::GetChildren(value);
		if (StructType::IsUnnamed(type)) {
			out += '[';
			for (idx_t i = 0; i < children.size(); i++) {
				if (i > 0) {
					out += ',';
				}
				AppendJson(out, children[i]);
			}
			out += ']';
			return;
		}
		out += '{';
		for (idx_t i = 0; i < children.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			AppendJsonString(out, StructType::GetChildName(type, i).GetIdentifierName());
			out += ':';
			AppendJson(out, children[i]);
		}
		out += '}';
		return;
	}
	case LogicalTypeId::MAP: {
		out += '{';
		auto &entries = MapValue::GetChildren(value);
		for (idx_t i = 0; i < entries.size(); i++) {
			if (i > 0) {
				out += ',';
			}
			auto &entry = StructValue::GetChildren(entries[i]);
			AppendJsonString(out, entry[0].ToString());
			out += ':';
			AppendJson(out, entry[1]);
		}
		out += '}';
		return;
	}
	default:
		AppendJsonString(out, value.ToString());
		return;
	}
}

static string CompactJson(const Value &value) {
	string out;
	AppendJson(out, value);
	return out;
}

//===--------------------------------------------------------------------===//
// State text
//===--------------------------------------------------------------------===//
//! A JSON document: a string as its text, anything else as compact JSON.
static string JsonStateText(const string &json) {
	auto doc = yyjson_read(json.data(), json.size(), YYJSON_READ_NOFLAG);
	if (!doc) {
		return json;
	}
	auto root = yyjson_doc_get_root(doc);
	string result;
	if (yyjson_is_str(root)) {
		result = string(yyjson_get_str(root), yyjson_get_len(root));
	} else {
		size_t length;
		auto text = yyjson_val_write(root, YYJSON_WRITE_NOFLAG, &length);
		result = text ? string(text, length) : json;
		free(text);
	}
	yyjson_doc_free(doc);
	return result;
}

string StateText(const Value &value) {
	if (value.IsNull()) {
		return string();
	}
	auto &type = value.type();
	if (type.IsJSONType()) {
		return JsonStateText(StringValue::Get(value));
	}
	if (IsTextType(type)) {
		return StringValue::Get(value.DefaultCastAs(LogicalType::VARCHAR));
	}
	if (IsNumericType(type) || type.IsNested()) {
		return CompactJson(value);
	}
	return value.ToString();
}

//===--------------------------------------------------------------------===//
// Options
//===--------------------------------------------------------------------===//
//! `opt_text` of the reference for a {key: description} entry: `key: description`, or the key alone.
static string DescribedOptionText(const string &key, const Value &description) {
	if (!description.IsNull()) {
		auto text = description.ToString();
		if (!text.empty()) {
			return key + ": " + text;
		}
	}
	return key;
}

vector<DecisionOption> DecisionOptions(const Value &options) {
	vector<DecisionOption> result;
	auto &type = options.type();
	switch (type.id()) {
	case LogicalTypeId::LIST:
		for (auto &option : ListValue::GetChildren(options)) {
			if (option.IsNull()) {
				throw InvalidInputException("decide_onnx: options must not contain NULL");
			}
			auto text = option.ToString();
			result.push_back({text, text});
		}
		break;
	case LogicalTypeId::MAP:
		for (auto &entry : MapValue::GetChildren(options)) {
			auto &key_value = StructValue::GetChildren(entry);
			auto key = key_value[0].ToString();
			result.push_back({key, DescribedOptionText(key, key_value[1])});
		}
		break;
	case LogicalTypeId::STRUCT: {
		auto &children = StructValue::GetChildren(options);
		for (idx_t i = 0; i < children.size(); i++) {
			auto key = StructType::GetChildName(type, i).GetIdentifierName();
			result.push_back({key, DescribedOptionText(key, children[i])});
		}
		break;
	}
	default:
		throw InternalException("decide_onnx: unexpected options type %s", type.ToString());
	}
	if (result.size() < 2) {
		throw InvalidInputException("decide_onnx: a decision needs at least 2 options, got %llu", result.size());
	}
	unordered_set<string> keys;
	for (auto &option : result) {
		if (option.key.empty()) {
			throw InvalidInputException("decide_onnx: options must not be empty");
		}
		if (!keys.insert(option.key).second) {
			throw InvalidInputException("decide_onnx: option '%s' is given more than once", option.key);
		}
	}
	return result;
}

} // namespace decide_onnx
} // namespace duckdb
