# vela3-fast ONNX export

These scripts turn the decision backbone of [vela3-fast](https://huggingface.co/vllm-sr/vela3-fast) into the ONNX
graph the `decide_onnx` DuckDB extension runs, and generate the parity fixtures the extension is tested against.
You only need them to cut a new asset version; users of the extension never run them.

```bash
cd scripts/export
uv sync
uv run python -I export_onnx.py    # out/, regenerates src/include/decide_onnx/model_manifest.hpp
uv run python make_fixtures.py     # test/data/parity/
```

Everything is pinned: the Hugging Face revision of vela3-fast (`HF_REVISION` in `export_onnx.py`) and the
torch/transformers/onnx/onnxruntime versions (`uv.lock`). The export downloads about 1.5 GB (`model.safetensors`
and the model's Python files); the span and router adapters are not needed.

## What gets exported

`DecisionEncoder` of `vela3_encoder.py` (the UltraBERT-380M encoder and its cosine + MLP readout) becomes one
graph for one prompt at a time. Its inputs and outputs are documented at the top of `export_onnx.py`. The graph
runs the published weights, but not the published forward code. That code builds dense `[T, T]` attention masks,
which needs about 7 GB at 8K tokens on a CPU. The export bounds that memory instead (linear in the prompt length
past 4,096 tokens), and maps each part of the network onto an ONNX Runtime kernel:
- The 8 global layers use `com.microsoft.MultiHeadAttention`. Its CPU kernel is a flash attention, but its GEMMs
  are plain NEON. Given a `key_padding_mask`, it computes the `[heads, T, T]` scores with ONNX Runtime's GEMM
  instead, which runs on KleidiAI's SME kernels on arm64 and is about twice as fast. So up to 4,096 tokens
  (`GEMM_ATTENTION_MAX`, 512 MB of scores), an `If` passes it an all-ones mask, which masks nothing. Longer prompts
  take the flash attention, whose memory is linear in the prompt length.
- The 16 sliding-window layers attend in blocks of 128 queries. Each block's keys are three slices of the padded
  keys, not a gather. A prompt of at most 129 tokens fits in one window, so an `If` sends it to
  `MultiHeadAttention` with the mask instead, which is exact and cheaper.
- RoPE uses `com.microsoft.RotaryEmbedding`, so q, k and v stay `[1, T, hidden]` and the global layers need no
  transposes.
- RMSNorm is written as `x / sqrt(mean(x^2) + eps)`, which ONNX Runtime fuses into
  `SimplifiedLayerNormalization`. It also fuses SiLU into `QuickGelu`.

The extension's ONNX Runtime build includes the contrib ops, and KleidiAI on arm64 for the matmuls. Compared with
a plain export, these choices cut a 30-token pass from about 61 ms to 33 ms on one Apple M4 Pro core
(`Transpose` alone was 21% of it). A short pass is now about 80% `MatMul`, which reads the 1.5 GB of fp32 weights;
a long one is mostly `MultiHeadAttention` and `MatMul`.

## Weight storage

The released `model.onnx` is the fp32 graph with its weights stored compactly, 452 MB instead of 1.52 GB:
- the 168 encoder projections as int8, symmetric, in blocks of 128 input rows, read back by a blocked
  `DequantizeLinear`, with the scales of ONNX Runtime's 8-bit `MatMulNBits` quantizer;
- the token embedding as fp16, read back by a `Cast` (an int8 embedding doubled the error);
- the readout head and the small tensors as fp32.

The extension creates its session with `session.disable_quant_qdq = 1`, so ONNX Runtime constant-folds these weights
back into fp32 when it loads the model and runs the same kernels as the fp32 graph, at the same speed. Without that
option it would fuse them into `MatMulNBits`, which multiplies int8 weights directly: less memory, but up to 8 times
slower on one core. The fp32 graph is kept as `out/model_fp32.onnx`.

Each export is checked against the unmodified PyTorch `DecisionEncoder` on 10 prompts of 10 to 9,006 tokens,
including 129 and 130 tokens on both sides of the `If`. A run fails if a logit of the fp32 graph is off by more than
1e-3 (the observed worst case is about 1e-5), or a logit of the released graph by more than 0.1 (about 0.04).

The prompt rendering (`ub_d3_common.to_row` and `DecisionEncoder.encode`) is ported to C++ in
`src/decision_model.cpp` and `src/state_text.cpp`. The fixtures come from the reference's own
`DecisionEncoder.predict`, so the parity tests cover that rendering as well as the graph.

## Determinism

Re-exporting does not give byte-identical files. Two runs produced different SHA-256s for `model.onnx`, though
both verify the same way. So the committed `model_manifest.hpp` and the uploaded files must come from the same
run: the extension refuses files whose SHA-256 differs from the header. The weight storage step is deterministic:
`export_onnx.py --skip-export` rebuilds the release from the `out/model_fp32.onnx` of an earlier run.

## Publishing a new asset version

1. Bump `ASSET_VERSION` in `export_onnx.py` if the graph changes, along with the default
   `decide_onnx_model_url` and `decide_onnx_cache_dir` in `src/model_store.cpp`.
2. Run both scripts and commit the regenerated header and fixtures.
3. Upload the release in `out/` to `https://assets.altertable.ai/decide_onnx/<ASSET_VERSION>/`: `model.onnx`,
   `tokenizer.json`, `manifest.json` and the weights' `LICENSE` (not `model_fp32.onnx`). The weights are Apache-2.0, and the licence must
   stay next to them.
