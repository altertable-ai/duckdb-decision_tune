#pragma once

#include <cstddef>

// This header, cpu_backend.cpp and the kernels that use them depend on neither DuckDB nor ONNX Runtime's headers, so
// that scripts/export/custom_ops builds them into the library the export checks its graphs with.

namespace duckdb {
namespace decide_onnx {

//! C = alpha * op(A) * op(B) + beta * C, row-major.
using Sgemm = void (*)(bool trans_a, bool trans_b, size_t m, size_t n, size_t k, float alpha, const float *a,
                       size_t lda, const float *b, size_t ldb, float beta, float *c, size_t ldc);
//! y[i] = exp(x[i]) for n values; x and y may be the same array.
using VecExp = void (*)(const float *x, float *y, size_t n);

//! The SGEMM and vector exp the custom kernels run on: Accelerate on Apple platforms, ONNX Runtime's MLAS elsewhere.
struct Backend {
	Sgemm gemm;
	VecExp exp;
};

//! The default backend of this build.
const Backend &DefaultBackend();

} // namespace decide_onnx
} // namespace duckdb
