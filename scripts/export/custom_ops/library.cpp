// The extension's custom ops (src/custom_ops.cpp) as a library for Python's onnxruntime, which loads it with
// SessionOptions.register_custom_ops_library. Built by build.sh, with ORT_API_MANUAL_INIT for every file.
#include "decide_onnx/custom_ops.hpp"

#include <onnxruntime_cxx_api.h>

extern "C" OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api) {
	Ort::InitApi(api->GetApi(ORT_API_VERSION));
	duckdb::decide_onnx::AddCustomOps(options);
	return nullptr;
}
