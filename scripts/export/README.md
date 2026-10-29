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
which needs about 7 GB at 8K tokens on a CPU. The export keeps memory linear in the prompt length instead:
- The 16 sliding-window layers attend in blocks of 128 queries.
- The 8 global layers use ONNX Runtime's `com.microsoft.MultiHeadAttention`, whose CPU kernel is a flash
  attention. The extension's ONNX Runtime build includes the contrib ops.

Before anything is written, each export is checked against the unmodified PyTorch `DecisionEncoder` on 9 prompts
of 10 to 9,006 tokens. A run fails if a logit is off by more than 1e-3. The observed worst case is about 1.3e-5.

The prompt rendering (`ub_d3_common.to_row` and `DecisionEncoder.encode`) is ported to C++ in
`src/decision_model.cpp` and `src/state_text.cpp`. The fixtures come from the reference's own
`DecisionEncoder.predict`, so the parity tests cover that rendering as well as the graph.

## Determinism

Re-exporting does not give byte-identical files. Two runs produced different SHA-256s for `model.onnx`, though
both verify the same way. So the committed `model_manifest.hpp` and the uploaded files must come from the same
run: the extension refuses files whose SHA-256 differs from the header.

## Publishing a new asset version

1. Bump `ASSET_VERSION` in `export_onnx.py` if the graph changes, along with the default
   `decide_onnx_model_url` and `decide_onnx_cache_dir` in `src/model_store.cpp`.
2. Run both scripts and commit the regenerated header and fixtures.
3. Upload everything in `out/` to `https://assets.altertable.ai/decide_onnx/<ASSET_VERSION>/`: `model.onnx`,
   `tokenizer.json`, `manifest.json` and the weights' `LICENSE`. The weights are Apache-2.0, and the licence must
   stay next to them.
