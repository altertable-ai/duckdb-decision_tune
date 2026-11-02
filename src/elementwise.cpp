#include "decide_onnx/elementwise.hpp"

#include <cmath>

namespace duckdb {
namespace decide_onnx {
namespace elementwise {

void SwiGlu(const float *x, int64_t rows, int64_t width, float *out, std::vector<float> &scratch,
            const Backend &backend) {
	scratch.resize(width);
	float *e = scratch.data();
	for (int64_t r = 0; r < rows; r++) {
		const float *gate = x + r * 2 * width, *up = gate + width;
		float *y = out + r * width;
		for (int64_t c = 0; c < width; c++) {
			e[c] = -gate[c];
		}
		backend.exp(e, e, width);
		for (int64_t c = 0; c < width; c++) {
			y[c] = gate[c] / (1.0f + e[c]) * up[c];
		}
	}
}

void AddRmsNorm(const float *x, const float *delta, const float *weight, int64_t rows, int64_t dim, float eps,
                float *sum, float *normed) {
	for (int64_t r = 0; r < rows; r++) {
		const float *xr = x + r * dim, *dr = delta + r * dim;
		float *s = sum + r * dim, *y = normed + r * dim;
		// Independent accumulators, so that the sum of squares vectorizes without -ffast-math.
		float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
		int64_t c = 0;
		for (; c + 8 <= dim; c += 8) {
			for (int64_t j = 0; j < 8; j++) {
				s[c + j] = xr[c + j] + dr[c + j];
				acc[j] += s[c + j] * s[c + j];
			}
		}
		float squares = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
		for (; c < dim; c++) {
			s[c] = xr[c] + dr[c];
			squares += s[c] * s[c];
		}
		const float scale = 1.0f / std::sqrt(squares / static_cast<float>(dim) + eps);
		for (c = 0; c < dim; c++) {
			y[c] = weight[c] * (s[c] * scale);
		}
	}
}

} // namespace elementwise
} // namespace decide_onnx
} // namespace duckdb
