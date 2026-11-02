# This file is included by DuckDB's build system. It specifies which extension to load

# Use one language standard for DuckDB and the extension. Mixing DuckDB's C++11
# out-of-line constexpr definitions with C++17 inline definitions can produce
# duplicate symbols on GCC (for example string_t::INLINE_LENGTH).
if(CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to enforce" FORCE)
endif()

duckdb_extension_load(decide_onnx
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)
