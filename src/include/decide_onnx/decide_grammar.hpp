#pragma once

namespace duckdb {
class DatabaseInstance;

namespace decide_onnx {

//! Registers the "decide_onnx" grammar extension (`FROM t DECIDE (...)`,
//! `decision_choose(question AMONG (...) GIVEN ...)` and `decision_yes_no(question GIVEN ...)`) and activates it on
//! every connection.
void RegisterDecideGrammar(DatabaseInstance &db);

} // namespace decide_onnx
} // namespace duckdb
