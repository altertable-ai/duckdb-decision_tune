# MLAS header

`mlas.h` is ONNX Runtime's `onnxruntime/core/mlas/inc/mlas.h` at tag `v1.29.0`, unmodified (MIT licence, © Microsoft
Corporation; see [LICENSE](LICENSE)). ONNX Runtime links MLAS, its linear algebra library, statically, but does not
install this header. `src/gated_delta_net.cpp` uses it on platforms other than Apple's, for `MlasGemm` and
`MlasComputeExp`.

It must match the ONNX Runtime version of `vcpkg_ports/onnxruntime`: replace it when that port changes version.
