#pragma once

namespace duckdb {
class ExtensionLoader;

namespace decide_onnx {

//! Registers `decision_choose(state, question, options)`, `decision_yes_no(state, question)` and
//! `decide_onnx_tokenize(text)`.
void RegisterDecideFunctions(ExtensionLoader &loader);

} // namespace decide_onnx
} // namespace duckdb
