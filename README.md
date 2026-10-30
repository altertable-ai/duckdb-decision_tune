# decide_onnx: decision models in DuckDB

`decide_onnx` runs [vela3-fast](https://huggingface.co/vllm-sr/vela3-fast) inside DuckDB. vela3-fast is the
0.4B member of the Vela 3.0 family from vLLM Semantic Router and KR Labs: an UltraBERT-380M encoder, Apache-2.0.
You give it a state, a question and options. It picks one option and gives a probability for each, or it gives
P(yes) for a yes/no question. It does one encoder pass and never generates text. The extension runs the model's
decision backbone only. The span, entity and router adapters of vela3-fast are not included.

Decision models of this kind are also offered as hosted APIs, such as TypeSafe's [Jev](https://typesafe.ai) and
OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions). vela3-fast is an open-weights
model that runs on your own machine. [Compared with hosted decision APIs](#compared-with-hosted-decision-apis)
explains the trade-offs.

With this extension, routing, tagging and triage are SQL queries. They need no Python, no API key and no GPU. The
network runs on CPU through a statically linked [ONNX Runtime](https://onnxruntime.ai/).

```sql
LOAD decide_onnx;

FROM tickets DECIDE (
    CHOOSE 'Which team should handle this customer message?' AMONG (
        billing  'Billing: charges, refunds, invoices',
        shipping 'Shipping: delivery, lost or damaged packages',
        tech     'Tech support: bugs, crashes, login problems') AS team,
    ASK 'Is the customer asking for a refund or an exchange?' AS refund
    GIVEN subject, message
);
```

| subject       | team     | team_confidence | team_probabilities                       | refund | refund_confidence | refund_probabilities |
|---------------|----------|----------------:|------------------------------------------|--------|------------------:|----------------------|
| Broken mug    | shipping | 0.72            | {billing=0.28, shipping=0.72, tech=0.01} | true   | 0.99              | {yes=0.99, no=0.01}  |
| Double charge | billing  | 1.00            | {billing=1.00, shipping=0.00, tech=0.00} | false  | 0.89              | {yes=0.11, no=0.89}  |
| Login loop    | tech     | 1.00            | {billing=0.00, shipping=0.00, tech=1.00} | false  | 0.94              | {yes=0.06, no=0.94}  |

## Install

`decide_onnx` is meant for DuckDB 2.0. Build it from source (see [Building](#building)), then load
`build/release/extension/decide_onnx/decide_onnx.duckdb_extension`. The draft for the community extensions
repository is in `description.yml`. Once that is published:

```sql
INSTALL decide_onnx FROM community;
LOAD decide_onnx;
```

`LOAD` adds the `DECIDE` syntax to every connection. A connection that was already open switches when its current
query ends, and this includes the connection that runs `LOAD`. So send `LOAD decide_onnx` and the first `DECIDE`
query as two separate statements, not as one string.

The first query that runs the model downloads vela3-fast from `https://assets.altertable.ai/decide_onnx/v4/`.
That is `tokenizer.json` (3.5 MB) and `model.onnx` (452 MB), an ONNX export of the decision backbone of one pinned
commit of the Hugging Face repository (see [scripts/export](scripts/export/README.md)). The file stores the weights
as int8 and fp16; ONNX Runtime turns them back into float32 when the model loads. The extension checks each
file against the SHA-256 compiled into it and caches it in `~/.duckdb/decide_onnx/v4/`. The model loads when a
query first needs an answer, not when the query is bound, so `DESCRIBE` of a `DECIDE` query stays fast. The
optimizer may compute a call whose arguments are all constants while it plans the query.

## Deciding

### `FROM … DECIDE (…)`

`DECIDE` follows a relation in `FROM`, as `PIVOT` does. It asks one or more questions about every row. It returns
the row's columns, then the answers:

```
FROM relation [alias]
DECIDE (
    CHOOSE question AMONG (key ['description'], ...) AS name
  | ASK question AS name
  [, ...]
    GIVEN expr [AS name] [, ...]
) [alias]
```

| Part | Meaning |
|---|---|
| `CHOOSE question AMONG (…) AS name` | Pick one of at least 2 options. A key is an identifier or a string. The model reads `key: description`, or the key alone when there is no description. |
| `ASK question AS name` | A yes/no question. |
| `AS name` | Names the answer columns. Required. |
| `GIVEN` | What the model reads: the *state*. With one expression, the state is its text. With several, it is a compact JSON object, `{"name":value,...}`, the way the reference reads a JSON state. Required. |

For each question, the clause adds three columns:

| Column | `CHOOSE` | `ASK` |
|---|---|---|
| `name` | The key of the most likely option (`VARCHAR`) | `true` when P(yes) ≥ 0.5 (`BOOLEAN`) |
| `name_confidence` | The probability of that option | `greatest(p_yes, 1 - p_yes)` |
| `name_probabilities` | Every option's probability, `MAP(VARCHAR, DOUBLE)` | `{'yes': p_yes, 'no': 1 - p_yes}` |

A `NULL` input gives `NULL` answers. The probabilities are what you threshold on, with an ordinary `WHERE`:

```sql
WITH triage AS (
    FROM tickets DECIDE (
        CHOOSE 'Which team should handle this customer message?' AMONG (
            billing 'Billing: charges, refunds, invoices',
            tech    'Tech support: bugs, crashes, login problems') AS team,
        ASK 'Is the customer asking for a refund?' AS refund
        GIVEN subject, message)
)
FROM triage
WHERE team_confidence < 0.6                     -- unsure routing: send to a person
   OR refund_probabilities['yes'] BETWEEN 0.2 AND 0.8;
```

DuckDB rewrites the clause into plain SQL around the functions below. So the clause works in CTEs, subqueries,
joins, prepared statements (`ASK $1`), views and table macros. A stored view keeps the rewritten SQL.

### `decision_choose(…)` and `decision_yes_no(…)`

The clause is built on these two functions. You can also call them yourself:

```sql
SELECT message,
       decision_choose(message, 'Which tool should be called?', {
           get_weather: 'Get the weather forecast for a city and date',
           send_email: 'Send an email to a person',
           none: 'No tool: answer the user directly'}).choice AS tool,
       decision_yes_no(message, 'Does this message ask to delete the account?') AS p_delete
FROM messages;
```

- `decision_choose(state, question, options)` returns
  `STRUCT(choice VARCHAR, confidence DOUBLE, probabilities MAP(VARCHAR, DOUBLE))`. The probabilities sum to 1.
  `options` can be any of these:
  - a list of texts, such as `['billing', 'shipping']`. Each text is its own key, and the model reads it as is.
  - a `MAP` of key to description, such as `MAP {'billing': 'charges, refunds, invoices'}`. The model reads
    `billing: charges, refunds, invoices`.
  - a `STRUCT` of key to description, such as `{billing: 'charges, refunds, invoices'}`, read the same way.
- `decision_yes_no(state, question)` returns P(yes) as a `DOUBLE`.

Both functions also accept the clause's words:

```sql
SELECT decision_choose('Which team?' AMONG (billing 'Billing: charges', tech 'Tech support: bugs') GIVEN subject, message),
       decision_yes_no('Is the customer asking for a refund?' GIVEN message)
FROM tickets;
```

The state can be any value. It is read the way the reference (`ub_d3_common.py`) reads a state:
- Text is read as is, and so is a JSON string.
- A `STRUCT`, a `MAP`, a list, a number, a boolean or any other JSON value is read as compact JSON, such as
  `{"subject":"Broken mug","n":3,"tags":["a","b"],"note":null}`.
- Dates, timestamps and UUIDs are read as their text.

A `NULL` state, question or options gives a `NULL` result. A prompt longer than the model's 32,768-token context
raises an error. The reference refuses such prompts too and does not truncate them. Shorten the input, for example
with `left(message, 20000)`, or wrap the call in `TRY(…)` to get `NULL` instead. Long prompts are slow on a CPU
(see [Performance](#performance)).

`decide_onnx_tokenize(text)` returns the token ids the model sees for a text. Use it to check lengths.

### Writing good questions

vela3-fast scores the options you give it. The quality of a decision depends on the question, the options and
the state. These tips help:
- **Describe every option.** `shipping 'delivery, lost or damaged packages'` usually gives better results
  than `shipping` alone.
- **Offer a way out**, such as `none 'None of these: spam, praise, or a message for no team'`.
- **Write down the criteria.** "Is this urgent?" alone gives weak results. Say what urgent means.
- **Ask one judgment per question.** Combine the answers in SQL.
- **Put only what a person would read into `GIVEN`.** Leave out ids and timestamps.
- **Phrase yes/no questions so that yes is the case you act on.** Then threshold `name_probabilities['yes']`.

vela3-fast was trained on English. The model's probabilities are not calibrated. Pick thresholds on your own
labeled rows, and send the unsure rows to a person.

## Performance

Every question for every row is one forward pass of the 380M encoder. The extension spreads the decisions of each
chunk of rows over DuckDB's worker threads, so `SET threads` sets the parallelism. ONNX Runtime itself runs
single-threaded. Measured on an Apple M4 Pro (10 performance and 4 efficiency cores), with tickets of about 20
tokens and one yes/no question each:

| Threads | Decisions per second |
|---|---|
| 1 | 20 (about 50 ms each) |
| 4 | 46 |
| 8 | 48 |
| 14 | 47 |

Throughput stops growing after a few threads. On Apple silicon the matrix multiplications run on the SME matrix
units, which each cluster of cores shares, so more threads wait for the same units. The ONNX graph scores one
prompt per pass, and each pass multiplies by all 1.5 GB of float32 weights, in blocks of 32 tokens: a 33-token
prompt costs about as much as a 64-token one. 10,000 rows with two questions therefore take about seven minutes.

Long prompts cost much more, because 8 of the 24 layers attend over the whole prompt. One prompt runs on one core:

| Prompt tokens | Time | Peak memory of the process |
|---|---|---|
| 2,000 | 1.8 s | 2.4 GB |
| 4,000 | 4.3 s | 3.1 GB |
| 8,000 | 29 s | 3.2 GB |
| 16,000 | 104 s | 3.9 GB |
| 32,000 | 6.5 min | 6.4 GB |

DuckDB only computes the decisions that a query uses. `SELECT count(*) FROM t DECIDE (...)` does not run the
model, and neither does a column that you do not select.

The extension keeps the probabilities of the last 100,000 distinct prompts in memory. A query that asks the same
question about the same state twice is scored once, and so is a query that you run again.

### Compared with hosted decision APIs

TypeSafe's Jev and OpenAI's Decisions API answer the same kind of question (pick an option, or give P(yes)) over
HTTP. The figures below are as published in October 2026. Only the vela3-fast throughput was measured here.

| | `decide_onnx` (vela3-fast) | [Jev](https://openrouter.ai/docs/guides/community/jev) (TypeSafe) | [Decisions API](https://developers.openai.com/api/docs/guides/decisions) (OpenAI) |
|---|---|---|---|
| Runs | In DuckDB, on your CPU | Hosted API | Hosted API (`gpt-6-luna`) |
| Weights | Open, Apache-2.0, 380M parameters | Closed | Closed |
| Price | No per-call cost | $0.042 per million input tokens, output free[^typesafe] | $0.10 per million input tokens, output free[^openai] |
| Speed | About 50 ms per decision on one core; 20 to 48 decisions per second on an Apple M4 Pro (above) | 0.114 s for an example workflow[^typesafe] | 100 to 180 ms per response[^techcrunch] |
| Decision Index, public index[^board] | 50.36 | 60.70 (Jev 1.14) | Not published |
| Decision Index, full score[^board] | 29.27 | 60.24 (Jev 1.14) | Not published |
| Context | 32,768 tokens | 262,144 tokens[^openrouter] | Not published |
| Your data | Stays on your machine | Sent to TypeSafe | Sent to OpenAI |

The [Decision Index](https://github.com/apolinario/decision-index) is chance-corrected: 0 is random guessing and
100 is perfect. The public index covers 37 public benchmarks. The full score is the board's ranking, which adds
private tests that only the board's maintainers run. Both rows come from the same board run. Jev scores 10.3 points
above vela3-fast on the public index and about twice its full score. vela3-fast has the highest public index of
the board's models up to 500M parameters, 21.5 points above DecisionTune 1.0 (28.85). OpenAI has published no
Decision Index score; TechCrunch reports that it ranks 14th on Artificial Analysis's Decision Index, where the
latest Jev ranks 3rd.[^techcrunch]

So the hosted models are likely more accurate, and at these prices 10,000 rows cost cents. Use this extension when the data
must not leave the machine, when the queries must work offline, when the result must be reproducible with a pinned
model, or when you want the decisions inside a SQL query without managing API keys, rate limits and retries. Check
the accuracy on a sample of your own labeled rows either way.

The speed figures do not compare like with like: the hosted figures include the network round trip and come from
the vendors, and a hosted API can answer many rows at once without your CPU or memory bandwidth.

[^typesafe]: [typesafe.ai](https://typesafe.ai): "$42 per billion input tokens"; example workflow completed in
    0.114 s.
[^openai]: [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions): `POST /v1/decisions`,
    up to 400 questions per request about one shared input.
[^techcrunch]: [TechCrunch, 22 October 2026](https://techcrunch.com/2026/10/22/openai-makes-its-jev-competitor-widely-available-as-decision-model-competition-heats-up/).
[^board]: [Decision Index board](https://huggingface.co/spaces/multimodalart/jev-decision-index), data of 29
    October 2026 (public panel `decision-index-0.3`). The vela3-fast numbers match its
    [model card](https://huggingface.co/vllm-sr/vela3-fast).
[^openrouter]: [OpenRouter Jev guide](https://openrouter.ai/docs/guides/community/jev). Requests over about 32,000
    bytes use a slower, less accurate long-context mode.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `decide_onnx_model_url` | `'https://assets.altertable.ai/decide_onnx/v4'` | The URL the files are downloaded from. A local directory also works. |
| `decide_onnx_cache_dir` | `'~/.duckdb/decide_onnx/v4'` | Where the downloaded files are cached. |
| `decide_onnx_model_path` | `''` | A local directory to use instead of the cache. It holds `tokenizer.json` and `model.onnx`. The files are not checksummed. |

### Offline use

`decide_onnx_models()` lists each file with its URL, SHA-256, size, local path and whether it is cached:

```sql
SELECT file_name, cached, local_path FROM decide_onnx_models();
```

To use the extension on a machine without internet access, do one of these:
- Copy `tokenizer.json` and `model.onnx` into the cache directory.
- Set `decide_onnx_model_path` to a directory that holds them, such as the `out/` directory of
  [scripts/export](scripts/export/README.md).

## How it works

- **Tokenizer.** vela3-fast uses UltraBERT's byte-level BPE tokenizer. `src/tokenizer.cpp` is a C++ port of the
  parts of Hugging Face `tokenizers` that this tokenizer uses:
  - added tokens, matched on the raw text, with their `lstrip` and `rstrip` flags
  - the GPT-2 pre-tokenizer regex
  - BPE merges

  It reads the published `tokenizer.json`. It also supports NFC normalization, which this tokenizer does not use.
- **Prompt.** `src/decision_model.cpp` builds the sequence the way `DecisionEncoder.encode` in `vela3_encoder.py`
  does: `<|endoftext|> [Q] question ([O] option)+ [SEP_SCHEMA] [SEG_user] state <|endoftext|>`. The markers are
  token ids after the tokenizer's vocabulary, so no input text can produce them. The ONNX graph returns one logit
  per `[O]` position, from the model's readout head (a cosine between the option and the pooled state, plus an
  MLP). A softmax over them gives the probabilities. A yes/no question is the choice between `no` and `yes`. The
  result has no calibration temperature, no option filtering and no truncation.
- **Graph.** `scripts/export/export_onnx.py` exports the decision backbone and checks it against the published
  PyTorch code. It bounds memory and uses ONNX Runtime's fused kernels: `com.microsoft.MultiHeadAttention` for
  the global layers and for short prompts (computing the scores with the GEMM up to 4,096 tokens, as a flash
  attention beyond), blocked attention for the sliding-window layers, `RotaryEmbedding` and
  `SimplifiedLayerNormalization`. See
  [scripts/export](scripts/export/README.md).
- **Syntax.** `DECIDE` and the `AMONG` / `GIVEN` call forms are additions to DuckDB's PEG grammar. They are
  registered as the `decide_onnx` grammar extension (`src/decide_grammar.cpp`). They only add rules and choices
  and replace no stock rule, so they combine with other grammar extensions (such as
  [timestral](https://github.com/altertable-ai/duckdb-timestral)). `SET active_grammar_extensions` turns them off
  for a connection.

The parity tests in `test/sql/parity_*.test` compare the extension with the published vela3-fast code, run in
PyTorch on the CPU (`scripts/export/make_fixtures.py`):
- Token ids match exactly on 394 cases. The cases cover multilingual text, emoji, combining marks, runs of spaces,
  contractions and special tokens.
- Probabilities match to within 0.02 on 14 decisions: choices with and without descriptions, yes/no questions, and
  text and JSON states. The largest difference is about 0.007, from the int8 storage of the weights; the same
  answer is chosen in every case.

## Limitations

- vela3-fast scores options. It does not reason step by step, and its probabilities are not calibrated. It was
  trained on English. Do not use it for high-risk decisions about people (hiring, credit, housing, health, legal)
  without independent validation.
- The ONNX graph takes one sequence at a time. Throughput comes from running many rows in parallel, not from
  batching.
- The extension is not available for DuckDB-WASM, Windows MinGW or Linux musl builds, because ONNX Runtime does
  not support them.
- The model needs about 2 GB of memory once it is loaded, and more while it reads a long prompt (see
  [Performance](#performance)). It is loaded once per process and shared by every connection.

## Building

```bash
git clone --recurse-submodules https://github.com/altertable-ai/duckdb-decide_onnx
cd duckdb-decide_onnx
git clone https://github.com/microsoft/vcpkg.git ../vcpkg && ../vcpkg/bootstrap-vcpkg.sh
export VCPKG_TOOLCHAIN_PATH=$PWD/../vcpkg/scripts/buildsystems/vcpkg.cmake
GEN=ninja make release
```

The build uses C++17 for both DuckDB and the extension. The first build compiles ONNX Runtime from source through
vcpkg, which takes a while. The overlay ports in `vcpkg_ports/` are the same as in duckdb-timestral, and
[`vcpkg_ports/README.md`](vcpkg_ports/README.md) explains them.

Run the tests:

```bash
make test
```

Tests that need the model are skipped unless you point them at a directory that holds `tokenizer.json` and
`model.onnx`. The extension's own cache directory works:

```bash
DECIDE_ONNX_MODEL_PATH=$HOME/.duckdb/decide_onnx/v4 make test
```

To export the model again or regenerate the parity fixtures, see [scripts/export](scripts/export/README.md).

## Licences

- This extension: MIT (see `LICENSE`).
- vela3-fast weights and tokenizer: Apache-2.0, vLLM Semantic Router and KR Labs. The ONNX graph on
  assets.altertable.ai is an export of those weights, under the same licence, which is published next to it.
- ONNX Runtime: MIT.
- vcpkg overlay ports, adapted from [anofox_tabfm](https://github.com/DataZooDE/anofox-tabfm): MIT.
