# decision_tune: decisions in DuckDB

`decision_tune` runs [DecisionTune 1.0](https://decisiontune.com/) inside DuckDB. DecisionTune is a 395M decision
model ([model card](https://huggingface.co/decision-tune/decisiontune-1.0),
[code](https://github.com/decision-tune/decision-tune)). You give it a state, a question and options. It picks one
option and gives a probability for each, or it gives P(yes) for a yes/no question. It does one encoder pass and
never generates text.

Decision models of this kind are also offered as hosted APIs, such as TypeSafe's [Jev](https://typesafe.ai) and
OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions). DecisionTune is an open-weights
model that runs on your own machine. [Compared with hosted decision APIs](#compared-with-hosted-decision-apis)
explains the trade-offs.

With this extension, routing, tagging and triage are SQL queries. They need no Python, no API key and no GPU. The
network runs on CPU through a statically linked [ONNX Runtime](https://onnxruntime.ai/).

```sql
LOAD decision_tune;

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
| Broken mug    | shipping | 0.78            | {billing=0.15, shipping=0.78, tech=0.07} | true   | 1.00              | {yes=1.00, no=0.00}  |
| Double charge | billing  | 0.73            | {billing=0.73, shipping=0.16, tech=0.11} | true   | 0.87              | {yes=0.87, no=0.13}  |
| Login loop    | tech     | 0.99            | {billing=0.00, shipping=0.00, tech=0.99} | false  | 0.98              | {yes=0.02, no=0.98}  |

## Install

`decision_tune` is meant for DuckDB 2.0. Build it from source (see [Building](#building)), then load
`build/release/extension/decision_tune/decision_tune.duckdb_extension`. The draft for the community extensions
repository is in `description.yml`. Once that is published:

```sql
INSTALL decision_tune FROM community;
LOAD decision_tune;
```

`LOAD` adds the `DECIDE` syntax to every connection. A connection that was already open switches when its current
query ends, and this includes the connection that runs `LOAD`. So send `LOAD decision_tune` and the first `DECIDE`
query as two separate statements, not as one string.

The first query that runs the model downloads DecisionTune 1.0 from Hugging Face. That is `tokenizer.json` (3.6 MB)
and `onnx/model.onnx` (1.58 GB), pinned to one repository commit. The extension checks each file against the
SHA-256 in the release's `manifest.json` and caches it in `~/.duckdb/decision_tune/1.0/`. The model loads when a
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
| `CHOOSE question AMONG (…) AS name` | Pick one of at least 2 options. A key is an identifier or a string. The description is what the model reads. Without a description, the model reads the key with `_` replaced by spaces. |
| `ASK question AS name` | A yes/no question. |
| `AS name` | Names the answer columns. Required. |
| `GIVEN` | What the model reads: the *state*. With one expression, the state is its text. With several, it is `name: value \| name: value`, the same as a decision-tune recipe that reads several columns. Required. |

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
  - a list of texts, such as `['billing', 'shipping']`. Each text is its own key.
  - a `MAP` of key to description, such as `MAP {'billing': 'Billing: charges, refunds'}`.
  - a `STRUCT` of key to description, such as `{billing: 'Billing: charges, refunds'}`.
- `decision_yes_no(state, question)` returns P(yes) as a `DOUBLE`.

Both functions also accept the clause's words:

```sql
SELECT decision_choose('Which team?' AMONG (billing 'Billing: charges', tech 'Tech support: bugs') GIVEN subject, message),
       decision_yes_no('Is the customer asking for a refund?' GIVEN message)
FROM tickets;
```

The state can be any value. It is read the way the reference engine reads it:
- Text is read as is.
- A `STRUCT`, a `MAP` or a JSON object is read as `key: value | key: value`. Text values are read as is, `NULL`
  as empty, and other values as compact JSON.
- Other values are read as compact JSON (`3`, `true`, `["a","b"]`). Dates, timestamps and UUIDs are read as their
  text.

A `NULL` state, question or options gives a `NULL` result. A prompt longer than the model's 8,192-token context
raises an error. The reference refuses such prompts too and does not truncate them. Shorten the input, for example
with `left(message, 20000)`, or wrap the call in `TRY(…)` to get `NULL` instead.

`decision_tune_tokenize(text)` returns the token ids the model sees for a text. Use it to check lengths.

### Writing good questions

DecisionTune scores the options you give it. The quality of a decision depends on the question, the options and
the state. Upstream's [prompting guide](https://github.com/decision-tune/decision-tune/blob/main/skills/decisiontune/references/prompting.md)
gives these tips:
- **Describe every option.** `shipping 'Shipping: delivery, lost or damaged packages'` gives much better results
  than `shipping` alone.
- **Offer a way out**, such as `none 'None of these: spam, praise, or a message for no team'`.
- **Write down the criteria.** "Is this urgent?" alone gives weak results. Say what urgent means.
- **Ask one judgment per question.** Combine the answers in SQL.
- **Put only what a person would read into `GIVEN`.** Leave out ids and timestamps.
- **Phrase yes/no questions so that yes is the case you act on.** Then threshold `name_probabilities['yes']`.

The model's probabilities are not calibrated. Pick thresholds on your own labeled rows, and send the unsure rows
to a person.

## Performance

Every question for every row is one forward pass of the 395M encoder. The extension spreads the decisions of each
chunk of rows over DuckDB's worker threads, so `SET threads` sets the parallelism. ONNX Runtime itself runs
single-threaded. Measured on an Apple M4 Pro (10 performance and 4 efficiency cores), with tickets of about 20
tokens and one yes/no question each:

| Threads | Decisions per second |
|---|---|
| 1 | 17 (about 60 ms each) |
| 4 | 59 |
| 8 | 57 |
| 14 | 79 |

Throughput stops growing after a few threads because memory bandwidth limits it, not compute. The published ONNX
graph scores one prompt per pass, and each pass reads all 1.6 GB of float32 weights. 10,000 rows with two
questions therefore take about four minutes.

DuckDB only computes the decisions that a query uses. `SELECT count(*) FROM t DECIDE (...)` does not run the
model, and neither does a column that you do not select.

The extension keeps the probabilities of the last 100,000 distinct prompts in memory. A query that asks the same
question about the same state twice is scored once, and so is a query that you run again.

### Compared with hosted decision APIs

TypeSafe's Jev and OpenAI's Decisions API answer the same kind of question (pick an option, or give P(yes)) over
HTTP. The figures below are as published in October 2026. Only the DecisionTune throughput was measured here.

| | `decision_tune` (DecisionTune 1.0) | [Jev](https://openrouter.ai/docs/guides/community/jev) (TypeSafe) | [Decisions API](https://developers.openai.com/api/docs/guides/decisions) (OpenAI) |
|---|---|---|---|
| Runs | In DuckDB, on your CPU | Hosted API | Hosted API (`gpt-6-luna`) |
| Weights | Open, Apache-2.0, 395M parameters | Closed | Closed |
| Price | No per-call cost | $0.042 per million input tokens, output free[^typesafe] | $0.10 per million input tokens, output free[^openai] |
| Speed | About 60 ms per decision on one core; 17 to 79 decisions per second on an Apple M4 Pro (above) | 0.114 s for an example workflow[^typesafe] | 100 to 180 ms per response[^techcrunch] |
| Decision Index 0.2.1 | 29.57[^card] | 57.89[^kit] | Not published |
| Context | 8,192 tokens | 262,144 tokens[^openrouter] | Not published |
| Your data | Stays on your machine | Sent to TypeSafe | Sent to OpenAI |

The [Decision Index](https://github.com/apolinario/decision-index) is chance-corrected: 0 is random guessing and
100 is perfect. On version 0.2.1, Jev scores about twice what DecisionTune 1.0 does. OpenAI has published no Decision
Index score; TechCrunch reports that it ranks 14th on Artificial Analysis's Decision Index, where the latest Jev
ranks 3rd.[^techcrunch]

So the hosted models are more accurate, and at these prices 10,000 rows cost cents. Use this extension when the data
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
[^card]: [DecisionTune 1.0 model card](https://huggingface.co/decision-tune/decisiontune-1.0): one complete run
    (raw 46.80); a second seed scored 29.13.
[^kit]: [Decision Index reproduction kit](https://github.com/apolinario/decision-index), reference score in its
    0.2.1 parity test.
[^openrouter]: [OpenRouter Jev guide](https://openrouter.ai/docs/guides/community/jev). Requests over about 32,000
    bytes use a slower, less accurate long-context mode.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `decision_tune_model_url` | `'https://huggingface.co/decision-tune/decisiontune-1.0/resolve/<commit>'` | The URL the files are downloaded from. A local directory also works. |
| `decision_tune_cache_dir` | `'~/.duckdb/decision_tune/1.0'` | Where the downloaded files are cached. |
| `decision_tune_model_path` | `''` | A local directory to use instead of the cache. It holds `tokenizer.json` and `model.onnx`, or is a clone of the Hugging Face repository. The files are not checksummed. |

### Offline use

`decision_tune_models()` lists each file with its URL, SHA-256, size, local path and whether it is cached:

```sql
SELECT file_name, cached, local_path FROM decision_tune_models();
```

To use the extension on a machine without internet access, do one of these:
- Copy `tokenizer.json` and `model.onnx` into the cache directory.
- Set `decision_tune_model_path` to a directory that holds them, such as a `git clone` of the model repository.

## How it works

- **Tokenizer.** DecisionTune uses ModernBERT's byte-level BPE tokenizer. `src/tokenizer.cpp` is a C++ port of
  the parts of Hugging Face `tokenizers` that this tokenizer uses:
  - added tokens, with special tokens matched on the raw text and the others after normalization
  - NFC normalization, through DuckDB's bundled utf8proc
  - the GPT-2 pre-tokenizer regex
  - BPE merges

  It reads the published `tokenizer.json`.
- **Prompt.** `src/decision_model.cpp` builds the sequence the way `engine.py` does:
  `[CLS] question [SEP] ([MASK] option)+ [SEP] state [SEP]`. It replaces special tokens in the input text with
  spaces. A softmax over the logits that the published ONNX graph returns for the `[MASK]` positions gives the
  probabilities. The result has no calibration temperature, no option filtering and no truncation.
- **Syntax.** `DECIDE` and the `AMONG` / `GIVEN` call forms are additions to DuckDB's PEG grammar. They are
  registered as the `decision_tune` grammar extension (`src/decide_grammar.cpp`). They only add rules and choices
  and replace no stock rule, so they combine with other grammar extensions (such as
  [timestral](https://github.com/altertable-ai/duckdb-timestral)). `SET active_grammar_extensions` turns them off
  for a connection.

The parity tests in `test/sql/parity_*.test` compare the extension with Hugging Face `tokenizers` and the Python
ONNX Runtime (`scripts/parity/make_fixtures.py`):
- Token ids match exactly on 392 cases. The cases cover multilingual text, emoji, combining marks, runs of spaces,
  contractions and special tokens. A further fuzz run over 20,000 random multilingual strings also matched.
- Probabilities match to within 1e-4. This includes the README example of the model card: Shipping 71.2%,
  Tech support 20.9%, Billing 7.9%.

## Limitations

- DecisionTune scores options. It does not reason step by step, and its probabilities are not calibrated. Do not
  use it for high-risk decisions about people (hiring, credit, housing, health, legal) without independent
  validation (see the model card).
- The ONNX graph takes one sequence at a time. Throughput comes from running many rows in parallel, not from
  batching.
- The extension is not available for DuckDB-WASM, Windows MinGW or Linux musl builds, because ONNX Runtime does
  not support them.
- The model needs about 1.6 GB of memory once it is loaded. It is loaded once per process and shared by every
  connection.

## Building

```bash
git clone --recurse-submodules https://github.com/altertable-ai/duckdb-decision_tune
cd duckdb-decision_tune
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
DECISION_TUNE_MODEL_PATH=$HOME/.duckdb/decision_tune/1.0 make test
```

To regenerate the parity fixtures:

```bash
cd scripts/parity && uv run make_fixtures.py ~/.duckdb/decision_tune/1.0
```

## Licences

- This extension: MIT (see `LICENSE`).
- DecisionTune 1.0 weights, ONNX graph and tokenizer: Apache-2.0, DecisionTune. The model card lists the
  licences and known issues of the training data.
- ONNX Runtime: MIT.
- vcpkg overlay ports, adapted from [anofox_tabfm](https://github.com/DataZooDE/anofox-tabfm): MIT.
