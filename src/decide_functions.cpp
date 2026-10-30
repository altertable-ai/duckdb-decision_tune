#include "decide_onnx/decide_functions.hpp"

#include "decide_onnx/model_store.hpp"
#include "decide_onnx/state_text.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/expression.hpp"

#include <functional>

namespace duckdb {
namespace decide_onnx {

//! One question about one state.
struct Decision {
	string state;
	string question;
	vector<string> options;
};

//===--------------------------------------------------------------------===//
// Scoring
//===--------------------------------------------------------------------===//
//! One of `count` independent steps (`step(index)`) as a DuckDB task.
class StepTask : public BaseExecutorTask {
public:
	StepTask(TaskExecutor &executor, ClientContext &context, const std::function<void(idx_t)> &step, idx_t index)
	    : BaseExecutorTask(executor), context(context), step(step), index(index) {
	}

	void ExecuteTask() override {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		step(index);
	}

	string TaskType() const override {
		return "DecideOnnxTask";
	}

private:
	ClientContext &context;
	const std::function<void(idx_t)> &step;
	idx_t index;
};

//! Runs `step(0)` .. `step(count - 1)` on DuckDB's scheduler, this thread included, and waits for them.
static void RunSteps(ClientContext &context, idx_t count, const std::function<void(idx_t)> &step) {
	TaskExecutor executor(context);
	for (idx_t i = 0; i < count; i++) {
		executor.ScheduleTask(make_uniq<StepTask>(executor, context, step, i));
	}
	executor.WorkOnTasks();
}

static string DecisionKey(const Decision &decision) {
	string key = decision.state;
	key += '\0';
	key += decision.question;
	for (auto &option : decision.options) {
		key += '\0';
		key += option;
	}
	return key;
}

//! The option probabilities of every decision. A chunk's distinct decisions run as tasks on DuckDB's scheduler: a
//! forward pass takes tens of milliseconds, and a small table is often read by a single thread. Short prompts share
//! forward passes (PlanPasses), which uses the CPU much better than one pass each.
static vector<std::vector<double>> ScoreDecisions(ClientContext &context, const vector<Decision> &decisions) {
	vector<std::vector<double>> result(decisions.size());
	if (decisions.empty()) {
		return result;
	}
	auto model = ModelStore::Instance().GetModel(context);

	vector<idx_t> unique_index(decisions.size());
	vector<idx_t> unique_decisions;
	unordered_map<string, idx_t> seen;
	for (idx_t i = 0; i < decisions.size(); i++) {
		auto entry = seen.emplace(DecisionKey(decisions[i]), unique_decisions.size());
		if (entry.second) {
			unique_decisions.push_back(i);
		}
		unique_index[i] = entry.first->second;
	}

	// Tokenizing a long state takes a while too, so it runs as tasks as well.
	vector<Prompt> prompts(unique_decisions.size());
	vector<std::vector<double>> unique_results(unique_decisions.size());
	vector<uint8_t> known(unique_decisions.size());
	RunSteps(context, unique_decisions.size(), [&](idx_t u) {
		auto &decision = decisions[unique_decisions[u]];
		prompts[u] = EncodePrompt(model->Tokenizer(), decision.state, decision.question, decision.options);
		known[u] = model->Lookup(prompts[u], unique_results[u]);
	});

	vector<idx_t> missing, lengths;
	for (idx_t u = 0; u < unique_decisions.size(); u++) {
		if (!known[u]) {
			missing.push_back(u);
			lengths.push_back(prompts[u].ids.size());
		}
	}
	auto passes = PlanPasses(lengths, TaskScheduler::GetScheduler(context).NumberOfThreads());
	RunSteps(context, passes.size(), [&](idx_t p) {
		vector<const Prompt *> pack;
		for (auto i : passes[p]) {
			pack.push_back(&prompts[missing[i]]);
		}
		auto probabilities = model->Probabilities(pack);
		for (idx_t i = 0; i < pack.size(); i++) {
			unique_results[missing[passes[p][i]]] = std::move(probabilities[i]);
		}
	});
	for (idx_t i = 0; i < decisions.size(); i++) {
		result[i] = unique_results[unique_index[i]];
	}
	return result;
}

//! The rows to compute: one if every argument is constant (the result is then constant too), else all of them.
static idx_t PrepareResult(DataChunk &args, Vector &result) {
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		return 1;
	}
	return args.size();
}

//===--------------------------------------------------------------------===//
// decision_choose(state, question, options)
//===--------------------------------------------------------------------===//
static LogicalType ChooseResultType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back(Identifier("choice"), LogicalType::VARCHAR);
	fields.emplace_back(Identifier("confidence"), LogicalType::DOUBLE);
	fields.emplace_back(Identifier("probabilities"), LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(fields));
}

static void ChooseResolveTypes(ResolveScalarFunctionTypesInput &input) {
	auto &bound_function = input.GetBoundFunction();
	auto &options_type = input.GetArguments()[2]->GetReturnType();
	auto &arguments = bound_function.GetArguments();
	switch (options_type.id()) {
	case LogicalTypeId::UNKNOWN:
		throw ParameterNotResolvedException();
	case LogicalTypeId::LIST:
	case LogicalTypeId::ARRAY:
		arguments[2] = LogicalType::LIST(LogicalType::VARCHAR);
		break;
	case LogicalTypeId::MAP:
		arguments[2] = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
		break;
	case LogicalTypeId::STRUCT: {
		child_list_t<LogicalType> fields;
		for (auto &child : StructType::GetChildTypes(options_type)) {
			fields.emplace_back(child.first, LogicalType::VARCHAR);
		}
		arguments[2] = LogicalType::STRUCT(std::move(fields));
		break;
	}
	default:
		throw BinderException("decision_choose: options must be a list of texts (['billing', 'shipping']), a MAP of "
		                      "key to description (MAP {'billing': 'Billing: charges, refunds'}) or a STRUCT "
		                      "({billing: 'Billing: charges, refunds'}), not %s",
		                      options_type.ToString());
	}
}

static void ChooseFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = PrepareResult(args, result);
	vector<Decision> decisions;
	vector<vector<DecisionOption>> options;
	vector<idx_t> rows;
	for (idx_t row = 0; row < count; row++) {
		auto state_value = args.data[0].GetValue(row);
		auto question = args.data[1].GetValue(row);
		auto options_value = args.data[2].GetValue(row);
		if (state_value.IsNull() || question.IsNull() || options_value.IsNull()) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		Decision decision {StateText(state_value), question.ToString(), {}};
		options.push_back(DecisionOptions(options_value));
		for (auto &option : options.back()) {
			decision.options.push_back(option.text);
		}
		decisions.push_back(std::move(decision));
		rows.push_back(row);
	}
	auto probabilities = ScoreDecisions(state.GetContext(), decisions);
	for (idx_t i = 0; i < decisions.size(); i++) {
		auto &p = probabilities[i];
		idx_t best = 0;
		vector<Value> keys, values;
		for (idx_t k = 0; k < p.size(); k++) {
			// The first option wins a tie, like max() over a dict.
			if (p[k] > p[best]) {
				best = k;
			}
			keys.emplace_back(options[i][k].key);
			values.push_back(Value::DOUBLE(p[k]));
		}
		result.SetValue(rows[i], Value::STRUCT(result.GetType(), {Value(options[i][best].key), Value::DOUBLE(p[best]),
		                                                          Value::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE,
		                                                                     std::move(keys), std::move(values))}));
	}
}

//===--------------------------------------------------------------------===//
// decision_yes_no(state, question)
//===--------------------------------------------------------------------===//
static void YesNoFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = PrepareResult(args, result);
	vector<Decision> decisions;
	vector<idx_t> rows;
	for (idx_t row = 0; row < count; row++) {
		auto state_value = args.data[0].GetValue(row);
		auto question = args.data[1].GetValue(row);
		if (state_value.IsNull() || question.IsNull()) {
			result.SetValue(row, Value(LogicalType::DOUBLE));
			continue;
		}
		// The reference asks a yes/no question as the choice {no, yes}.
		decisions.push_back(Decision {StateText(state_value), question.ToString(), {"no", "yes"}});
		rows.push_back(row);
	}
	auto probabilities = ScoreDecisions(state.GetContext(), decisions);
	for (idx_t i = 0; i < decisions.size(); i++) {
		result.SetValue(rows[i], Value::DOUBLE(probabilities[i][1]));
	}
}

//===--------------------------------------------------------------------===//
// decide_onnx_tokenize(text)
//===--------------------------------------------------------------------===//
static void TokenizeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = PrepareResult(args, result);
	shared_ptr<BpeTokenizer> tokenizer;
	for (idx_t row = 0; row < count; row++) {
		auto text = args.data[0].GetValue(row);
		if (text.IsNull()) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		if (!tokenizer) {
			tokenizer = ModelStore::Instance().GetTokenizer(state.GetContext());
		}
		vector<Value> ids;
		for (auto id : tokenizer->Encode(StringValue::Get(text))) {
			ids.push_back(Value::INTEGER(static_cast<int32_t>(id)));
		}
		result.SetValue(row, Value::LIST(LogicalType::INTEGER, std::move(ids)));
	}
}

void RegisterDecideFunctions(ExtensionLoader &loader) {
	ScalarFunction choose("decision_choose", {}, ChooseResultType(), ChooseFunction);
	choose.GetSignature()
	    .AddParameter("state", LogicalType::ANY)
	    .AddParameter("question", LogicalType::VARCHAR)
	    .AddParameter("options", LogicalType::ANY);
	choose.SetResolveTypesCallback(ChooseResolveTypes);
	choose.SetFallible();
	loader.RegisterFunction(choose);

	ScalarFunction yes_no("decision_yes_no", {}, LogicalType::DOUBLE, YesNoFunction);
	yes_no.GetSignature().AddParameter("state", LogicalType::ANY).AddParameter("question", LogicalType::VARCHAR);
	yes_no.SetFallible();
	loader.RegisterFunction(yes_no);

	ScalarFunction tokenize("decide_onnx_tokenize", {}, LogicalType::LIST(LogicalType::INTEGER), TokenizeFunction);
	tokenize.GetSignature().AddParameter("text", LogicalType::VARCHAR);
	loader.RegisterFunction(tokenize);
}

} // namespace decide_onnx
} // namespace duckdb
