#pragma once

#include "duckdb/common/types/value.hpp"

namespace duckdb {
namespace decide_onnx {

//! The text the model reads for a state, as `text` of the reference (ub_d3_common.py) renders it: text as is; a
//! STRUCT, a MAP, a LIST, a number, a boolean or a JSON value as compact JSON (a JSON string as its text); any other
//! value (dates, timestamps and the like) as its text.
string StateText(const Value &value);

//! One option of a decision: the key returned to the caller and the text the model reads.
struct DecisionOption {
	string key;
	string text;
};

//! The options of `decision_choose`: a LIST of texts (each its own key, read as is), or a MAP or STRUCT of key ->
//! description (read as `key: description`, or the key alone when the description is empty or NULL). Throws
//! InvalidInputException unless there are at least two distinct, non-empty keys.
vector<DecisionOption> DecisionOptions(const Value &options);

} // namespace decide_onnx
} // namespace duckdb
