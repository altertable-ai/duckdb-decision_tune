#define DUCKDB_EXTENSION_MAIN

#include "decide_onnx_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "decide_onnx/decide_functions.hpp"
#include "decide_onnx/decide_grammar.hpp"
#include "decide_onnx/model_manifest.hpp"
#include "decide_onnx/model_store.hpp"
#include "decide_onnx/scorer.hpp"

namespace duckdb {

static void DecideOnnxVersionFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &children = StructVector::GetEntries(result);
	DecideOnnxExtension extension;
	auto extension_version = extension.Version();
	children[0].SetValue(0, Value(extension_version.empty() ? "dev" : extension_version));
	children[1].SetValue(0, Value(decide_onnx::MODEL_NAME));
	children[2].SetValue(0, Value(decide_onnx::OrtScorer::RuntimeVersion()));
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
}

static void LoadInternal(ExtensionLoader &loader) {
	auto version_type = LogicalType::STRUCT(
	    {{"extension", LogicalType::VARCHAR}, {"model", LogicalType::VARCHAR}, {"onnxruntime", LogicalType::VARCHAR}});
	ScalarFunction version("decide_onnx_version", {}, version_type, DecideOnnxVersionFun);
	version.SetStability(FunctionStability::CONSISTENT);
	loader.RegisterFunction(version);

	decide_onnx::RegisterModelStore(loader);
	decide_onnx::RegisterDecideFunctions(loader);
	decide_onnx::RegisterDecideGrammar(loader.GetDatabaseInstance());
}

void DecideOnnxExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string DecideOnnxExtension::Name() {
	return "decide_onnx";
}

std::string DecideOnnxExtension::Version() const {
#ifdef EXT_VERSION_DECIDE_ONNX
	return EXT_VERSION_DECIDE_ONNX;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(decide_onnx, loader) {
	duckdb::LoadInternal(loader);
}
}
