cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME decide_onnx)

# ONNX Runtime (CPU execution provider) from vcpkg, linked statically. See
# vcpkg_ports/README.md. DuckDB's tidy configure is not a vcpkg build, so the
# package is optional there.
if(CLANG_TIDY)
  find_package(onnxruntime CONFIG QUIET)
else()
  find_package(onnxruntime CONFIG REQUIRED)
endif()
# vcpkg installs the public headers under include/onnxruntime/; sources include
# them flat.
if(onnxruntime_FOUND)
  find_path(
    DECIDE_ONNX_ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime REQUIRED)
endif()

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

include_directories(src/include)

set(EXTENSION_SOURCES
    src/decide_onnx_extension.cpp
    src/decide_functions.cpp
    src/decide_grammar.cpp
    src/decision_model.cpp
    src/model_store.cpp
    src/state_text.cpp
    src/tokenizer.cpp)
# This file includes onnxruntime headers. Skip it when the tidy job has no ONNX
# Runtime.
if(onnxruntime_FOUND)
  list(APPEND EXTENSION_SOURCES src/ort_scorer.cpp)
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

foreach(_target ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
  target_compile_features(${_target} PUBLIC cxx_std_17)
endforeach()

if(onnxruntime_FOUND)
  foreach(_target ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    target_include_directories(${_target}
                               PRIVATE ${DECIDE_ONNX_ORT_INCLUDE_DIR})
    target_link_libraries(${_target} onnxruntime::onnxruntime)
  endforeach()
endif()
# SHA-256 comes from DuckDB's bundled mbedtls (as in httpfs), JSON parsing from
# its bundled yyjson and Unicode data from its bundled utf8proc; the static
# extension gets them through duckdb_static.
target_link_libraries(${LOADABLE_EXTENSION_NAME} duckdb_mbedtls duckdb_yyjson
                      duckdb_utf8proc)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
