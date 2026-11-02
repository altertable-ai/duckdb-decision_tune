#pragma once

#include "decide_onnx/cpu_backend.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {
namespace decide_onnx {
namespace elementwise {

//! out [rows, width] = SiLU(gate) * up, for x [rows, 2 * width] holding gate then up on each row: the activation of
//! Qwen3.5's MLP over its fused gate and up projections. `scratch` is resized to one row.
void SwiGlu(const float *x, int64_t rows, int64_t width, float *out, std::vector<float> &scratch,
            const Backend &backend);

//! sum = x + delta and normed = weight * RMSNorm(sum), row by row, for x and delta [rows, dim]: a residual connection
//! and the RMSNorm after it. `sum` may be `x`.
void AddRmsNorm(const float *x, const float *delta, const float *weight, int64_t rows, int64_t dim, float eps,
                float *sum, float *normed);

} // namespace elementwise
} // namespace decide_onnx
} // namespace duckdb
