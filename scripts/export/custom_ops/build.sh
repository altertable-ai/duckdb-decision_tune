#!/usr/bin/env sh
# Builds out/libdecide_onnx_ops.{dylib,so}: the extension's custom ops (src/custom_ops.cpp and the kernels it runs) for
# Python's onnxruntime, so that export_onejev.py --custom-ops can check the graphs that use them.
#
#   sh custom_ops/build.sh            (from scripts/export)
#
# It takes ONNX Runtime's headers, and off Apple platforms its MLAS libraries, from the extension's vcpkg build
# (`make release`), which VCPKG_INSTALLED points at (default: build/release/vcpkg_installed/<the only triplet>).
# MLAS=1 builds the MLAS backend on Apple platforms too, as out/libdecide_onnx_ops_mlas.dylib.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
if [ -z "${VCPKG_INSTALLED:-}" ]; then
	VCPKG_INSTALLED=$(ls -d "$root"/build/release/vcpkg_installed/*/ | grep -v vcpkg/ | head -n 1)
fi
mkdir -p "$here/../out"
sources="$here/library.cpp $root/src/custom_ops.cpp $root/src/cpu_backend.cpp $root/src/elementwise.cpp
	$root/src/causal_attention.cpp $root/src/gated_delta_net.cpp"
flags="-std=c++17 -O3 -shared -fPIC -DORT_API_MANUAL_INIT -I$root/src/include -I$VCPKG_INSTALLED/include/onnxruntime"
lib="$VCPKG_INSTALLED/lib"
out="$here/../out/libdecide_onnx_ops"
case "$(uname)" in
Darwin)
	if [ "${MLAS:-0}" = 1 ]; then
		c++ $flags -DDECIDE_ONNX_USE_MLAS -I"$root/third_party/mlas" $sources "$lib/libonnxruntime_mlas.a" \
			"$lib/libonnxruntime_common.a" "$lib/libcpuinfo.a" "$lib/libkleidiai.a" "$lib"/libabsl_*.a \
			-o "${out}_mlas.dylib"
	else
		c++ $flags $sources -framework Accelerate -o "$out.dylib"
	fi
	;;
*)
	c++ $flags -I"$root/third_party/mlas" $sources -Wl,--start-group "$lib/libonnxruntime_mlas.a" \
		"$lib/libonnxruntime_common.a" "$lib/libcpuinfo.a" $(ls "$lib"/libkleidiai.a 2>/dev/null || true) \
		"$lib"/libabsl_*.a -Wl,--end-group -o "$out.so"
	;;
esac
echo "built $(ls -t "$out"* | head -n 1)"
