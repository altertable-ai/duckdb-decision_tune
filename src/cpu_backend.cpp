#include "decide_onnx/cpu_backend.hpp"

// DECIDE_ONNX_USE_MLAS selects MLAS on Apple platforms too, to test that path there.
#if defined(__APPLE__) && !defined(DECIDE_ONNX_USE_MLAS)
#define DECIDE_ONNX_ACCELERATE
#endif

#ifdef DECIDE_ONNX_ACCELERATE
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#else
// MLAS, ONNX Runtime's linear algebra library, is linked statically with ONNX Runtime; its header is not installed, so
// third_party/mlas carries the one of the ONNX Runtime version the vcpkg port builds.
#include "mlas.h"
#endif

namespace duckdb {
namespace decide_onnx {

#ifdef DECIDE_ONNX_ACCELERATE
static void Gemm(bool trans_a, bool trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda,
                 const float *b, size_t ldb, float beta, float *c, size_t ldc) {
	cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
	            static_cast<__LAPACK_int>(m), static_cast<__LAPACK_int>(n), static_cast<__LAPACK_int>(k), alpha, a,
	            static_cast<__LAPACK_int>(lda), b, static_cast<__LAPACK_int>(ldb), beta, c,
	            static_cast<__LAPACK_int>(ldc));
}

static void Exp(const float *x, float *y, size_t n) {
	const int count = static_cast<int>(n);
	vvexpf(y, x, &count);
}
#else
static void Gemm(bool trans_a, bool trans_b, size_t m, size_t n, size_t k, float alpha, const float *a, size_t lda,
                 const float *b, size_t ldb, float beta, float *c, size_t ldc) {
	MlasGemm(trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb,
	         beta, c, ldc, nullptr, nullptr);
}

static void Exp(const float *x, float *y, size_t n) {
	MlasComputeExp(x, y, n);
}
#endif

const Backend &DefaultBackend() {
	static const Backend backend {Gemm, Exp};
	return backend;
}

} // namespace decide_onnx
} // namespace duckdb
