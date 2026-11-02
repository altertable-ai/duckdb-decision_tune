#include "decide_onnx/decide_grammar.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection_manager.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/operator_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/grammar_extension.hpp"
#include "duckdb/parser/peg/compiled_grammar.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/planner/extension_callback.hpp"

namespace duckdb {
namespace decide_onnx {

//===--------------------------------------------------------------------===//
// Grammar
//===--------------------------------------------------------------------===//
static const char *const DECIDE_RULES[] = {
    // FROM <relation> DECIDE (...), tried before the stock table refs. Without an alias of its own the relation
    // reads DECIDE as its alias, hence the optional keyword.
    "DecideTableRef <- DecideSource DecideKeyword? Parens(DecideBody) TableAlias?",
    "DecideSource <- ValuesRef / TableFunction / TableSubquery / BaseTableRef / ParensTableRef",
    "DecideKeyword <- 'DECIDE'",
    "DecideBody <- List(DecideQuestion) DecideGiven?",
    "DecideQuestion <- (DecideChoose / DecideAsk) DecideQuestionAlias?",
    "DecideQuestionAlias <- 'AS' ColLabelOrString",
    "DecideChoose <- 'CHOOSE' Expression DecideAmong",
    "DecideAsk <- 'ASK' Expression",
    "DecideAmong <- 'AMONG' Parens(List(DecideOption))",
    "DecideOption <- ColIdOrString Expression?",
    "DecideGiven <- 'GIVEN' List(DecideInput)",
    "DecideInput <- Expression DecideInputAlias?",
    "DecideInputAlias <- 'AS' ColLabelOrString",
    // decision_choose(question AMONG (...) GIVEN ...) and decision_yes_no(question GIVEN ...): the functions,
    // spelled like the clause
    "DecideChooseCall <- 'DECISION_CHOOSE' Parens(DecideChooseCallBody)",
    "DecideChooseCallBody <- Expression DecideAmong DecideGiven",
    "DecideYesNoCall <- 'DECISION_YES_NO' Parens(DecideYesNoCallBody)",
    "DecideYesNoCallBody <- Expression DecideGiven",
};

static constexpr const char *DECISION_COLUMN_PREFIX = "__decision_";

//===--------------------------------------------------------------------===//
// Parse result helpers
//===--------------------------------------------------------------------===//
template <class T>
using rule_transform_t = T (*)(PEGTransformer &transformer, ParseResult &parse_result);

template <class T>
static grammar_transform_process_function_t MakeTransform(rule_transform_t<T> transform) {
	return [transform](PEGTransformer &transformer, ParseResult &parse_result) {
		return transformer.MakeProcess<FinalizeTransformProcess>(
		    transformer, parse_result, [transform](PEGTransformer &inner, ParseResult &inner_result) {
			    return inner.MakeResult<T>(transform(inner, inner_result));
		    });
	};
}

static optional_ptr<ParseResult> OptionalChild(ListParseResult &list, idx_t child_idx) {
	auto &opt = list.Child<OptionalParseResult>(child_idx);
	if (!opt.HasResult()) {
		return nullptr;
	}
	return &opt.GetResult();
}

static ParseResult &SelectedChoice(ParseResult &parse_result) {
	optional_ptr<ParseResult> current = &parse_result;
	while (true) {
		if (current->type == ParseResultType::CHOICE) {
			current = &current->Cast<ChoiceParseResult>().GetResult();
			continue;
		}
		if (current->type == ParseResultType::LIST) {
			auto &list = current->Cast<ListParseResult>();
			if (list.GetChildren().size() == 1 && list.GetChild(0).type == ParseResultType::CHOICE) {
				current = &list.GetChild(0);
				continue;
			}
		}
		return *current;
	}
}

static unique_ptr<ParsedExpression> TransformExpression(PEGTransformer &transformer, ParseResult &parse_result) {
	return transformer.Transform<unique_ptr<ParsedExpression>>(parse_result);
}

static string TransformName(PEGTransformer &transformer, ParseResult &parse_result) {
	return transformer.Transform<Identifier>(parse_result).GetIdentifierName();
}

static vector<reference<ParseResult>> ListEntries(ParseResult &parse_result) {
	return PEGTransformerFactory::ExtractParseResultsFromList(parse_result);
}

//===--------------------------------------------------------------------===//
// Specification
//===--------------------------------------------------------------------===//
struct DecideOptionSpec {
	string key;
	//! nullptr: the key is the text
	unique_ptr<ParsedExpression> description;
};

struct DecideQuestionSpec {
	string name;
	bool yes_no = false;
	unique_ptr<ParsedExpression> question;
	vector<DecideOptionSpec> options;
};

struct DecideInputSpec {
	unique_ptr<ParsedExpression> expression;
	string name;
};

struct DecideSpec {
	vector<DecideQuestionSpec> questions;
	vector<DecideInputSpec> inputs;
};

// DecideAmong <- 'AMONG' Parens(List(DecideOption))
// DecideOption <- ColIdOrString Expression?
static vector<DecideOptionSpec> TransformAmong(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &options_pr = PEGTransformerFactory::ExtractResultFromParens(parse_result.Cast<ListParseResult>().GetChild(1));
	vector<DecideOptionSpec> options;
	unordered_set<string> keys;
	for (auto &entry : ListEntries(options_pr)) {
		auto &option_pr = entry.get().Cast<ListParseResult>();
		DecideOptionSpec option;
		option.key = TransformName(transformer, option_pr.GetChild(0));
		if (auto description = OptionalChild(option_pr, 1)) {
			option.description = TransformExpression(transformer, *description);
		}
		if (option.key.empty()) {
			throw ParserException("DECIDE: an option must not be empty");
		}
		if (!keys.insert(option.key).second) {
			throw ParserException("DECIDE: option '%s' is given more than once", option.key);
		}
		options.push_back(std::move(option));
	}
	if (options.size() < 2) {
		throw ParserException("DECIDE: AMONG needs at least 2 options");
	}
	return options;
}

// DecideGiven <- 'GIVEN' List(DecideInput)
// DecideInput <- Expression DecideInputAlias?
static vector<DecideInputSpec> TransformGiven(PEGTransformer &transformer, ParseResult &parse_result) {
	vector<DecideInputSpec> inputs;
	unordered_set<string> names;
	for (auto &entry : ListEntries(parse_result.Cast<ListParseResult>().GetChild(1))) {
		auto &input_pr = entry.get().Cast<ListParseResult>();
		DecideInputSpec input;
		input.expression = TransformExpression(transformer, input_pr.GetChild(0));
		if (auto alias = OptionalChild(input_pr, 1)) {
			// DecideInputAlias <- 'AS' ColLabelOrString
			input.name = TransformName(transformer, alias->Cast<ListParseResult>().GetChild(1));
		} else if (input.expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
			input.name = input.expression->Cast<ColumnRefExpression>().GetColumnName().GetIdentifierName();
		} else {
			input.name = input.expression->ToString();
		}
		if (!names.insert(StringUtil::Lower(input.name)).second) {
			throw ParserException("DECIDE: GIVEN reads '%s' more than once; name the inputs with AS", input.name);
		}
		inputs.push_back(std::move(input));
	}
	return inputs;
}

// DecideQuestion <- (DecideChoose / DecideAsk) DecideQuestionAlias?
// DecideQuestionAlias <- 'AS' ColLabelOrString
static DecideQuestionSpec TransformQuestion(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	DecideQuestionSpec spec;
	auto &kind = SelectedChoice(list_pr.GetChild(0));
	auto &kind_pr = kind.Cast<ListParseResult>();
	spec.question = TransformExpression(transformer, kind_pr.GetChild(1));
	if (StringUtil::CIEquals(kind.Name(), "DecideAsk")) {
		spec.yes_no = true;
	} else if (StringUtil::CIEquals(kind.Name(), "DecideChoose")) {
		spec.options = TransformAmong(transformer, kind_pr.GetChild(2));
	} else {
		throw InternalException("decide_onnx: unexpected DECIDE question rule '%s'", kind.Name());
	}
	// The name is optional in the grammar only to give a clearer error than a syntax error.
	auto alias = OptionalChild(list_pr, 1);
	if (!alias) {
		throw ParserException("DECIDE: name the answer of every question with AS, e.g. %s %s AS answer",
		                      spec.yes_no ? "ASK" : "CHOOSE", spec.question->ToString());
	}
	spec.name = TransformName(transformer, alias->Cast<ListParseResult>().GetChild(1));
	return spec;
}

// DecideBody <- List(DecideQuestion) DecideGiven?
static DecideSpec TransformDecideBody(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	DecideSpec spec;
	unordered_set<string> columns;
	for (auto &entry : ListEntries(list_pr.GetChild(0))) {
		auto question = TransformQuestion(transformer, entry.get());
		for (auto &output : {question.name, question.name + "_confidence", question.name + "_probabilities"}) {
			if (!columns.insert(StringUtil::Lower(output)).second) {
				throw ParserException("DECIDE: question '%s' would add a second column named '%s'", question.name,
				                      output);
			}
		}
		spec.questions.push_back(std::move(question));
	}
	// GIVEN is optional in the grammar only to give a clearer error than a syntax error.
	auto given = OptionalChild(list_pr, 1);
	if (!given) {
		throw ParserException("DECIDE requires GIVEN, the columns the model reads (e.g. GIVEN subject, message)");
	}
	spec.inputs = TransformGiven(transformer, *given);
	return spec;
}

//===--------------------------------------------------------------------===//
// Lowering
//===--------------------------------------------------------------------===//
static unique_ptr<ParsedExpression> WithAlias(unique_ptr<ParsedExpression> expr, const string &alias) {
	expr->SetAlias(Identifier(alias));
	return expr;
}

static unique_ptr<ParsedExpression> Column(const string &name) {
	return make_uniq<ColumnRefExpression>(Identifier(name));
}

static unique_ptr<ParsedExpression> Function(const string &name, vector<unique_ptr<ParsedExpression>> children) {
	return make_uniq<FunctionExpression>(Identifier(name), std::move(children));
}

template <class... ARGS>
static unique_ptr<ParsedExpression> Function(const string &name, ARGS... args) {
	vector<unique_ptr<ParsedExpression>> children;
	(void)std::initializer_list<int> {(children.push_back(std::move(args)), 0)...};
	return Function(name, std::move(children));
}

//! The state: the single input itself, or struct_pack(name := input, ...) read as a compact JSON object.
static unique_ptr<ParsedExpression> StateExpression(const vector<DecideInputSpec> &inputs) {
	if (inputs.size() == 1) {
		return inputs[0].expression->Copy();
	}
	vector<FunctionArgument> fields;
	for (auto &input : inputs) {
		fields.emplace_back(Identifier(input.name), WithAlias(input.expression->Copy(), input.name));
	}
	return make_uniq<FunctionExpression>(Identifier("struct_pack"), std::move(fields));
}

//! map(['billing', ...], ['Billing: charges, refunds', NULL, ...]); a NULL description reads as the key.
static unique_ptr<ParsedExpression> OptionsExpression(const vector<DecideOptionSpec> &options) {
	vector<unique_ptr<ParsedExpression>> keys, descriptions;
	for (auto &option : options) {
		keys.push_back(ConstantExpression::String(option.key));
		descriptions.push_back(option.description
		                           ? option.description->Copy()
		                           : make_uniq<CastExpression>(LogicalType::VARCHAR, ConstantExpression::Null()));
	}
	return Function("map", Function("list_value", std::move(keys)), Function("list_value", std::move(descriptions)));
}

static unique_ptr<ParsedExpression> ChooseCall(unique_ptr<ParsedExpression> state, const DecideQuestionSpec &question) {
	return Function("decision_choose", std::move(state), question.question->Copy(),
	                OptionsExpression(question.options));
}

static unique_ptr<ParsedExpression> YesNoCall(unique_ptr<ParsedExpression> state, const DecideQuestionSpec &question) {
	return Function("decision_yes_no", std::move(state), question.question->Copy());
}

static unique_ptr<SelectStatement> MakeStatement(unique_ptr<QueryNode> node) {
	auto statement = make_uniq<SelectStatement>();
	statement->node = std::move(node);
	return statement;
}

//! FROM <source> DECIDE (CHOOSE ... AS team, ASK ... AS refund GIVEN ...) becomes
//! SELECT * EXCLUDE (__decision_0, __decision_1),
//!        __decision_0.choice AS team, __decision_0.confidence AS team_confidence,
//!        __decision_0.probabilities AS team_probabilities,
//!        __decision_1 >= 0.5 AS refund, greatest(__decision_1, 1 - __decision_1) AS refund_confidence,
//!        map(['yes', 'no'], [__decision_1, 1 - __decision_1]) AS refund_probabilities
//! FROM (SELECT *, decision_choose(<state>, ...) AS __decision_0, decision_yes_no(<state>, ...) AS __decision_1
//!       FROM <source>)
static unique_ptr<SelectNode> LowerDecide(const DecideSpec &spec, unique_ptr<TableRef> source) {
	auto inner = make_uniq<SelectNode>();
	inner->select_list.push_back(make_uniq<StarExpression>());
	auto outer = make_uniq<SelectNode>();
	auto star = make_uniq<StarExpression>();
	vector<unique_ptr<ParsedExpression>> outputs;
	auto probabilities_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE);
	for (idx_t i = 0; i < spec.questions.size(); i++) {
		auto &question = spec.questions[i];
		auto decision = DECISION_COLUMN_PREFIX + to_string(i);
		auto state = StateExpression(spec.inputs);
		inner->select_list.push_back(
		    WithAlias(question.yes_no ? YesNoCall(std::move(state), question) : ChooseCall(std::move(state), question),
		              decision));
		star->ExcludeListMutable().insert(QualifiedColumnName(Identifier(decision)));

		if (question.yes_no) {
			auto p_no = [&]() {
				return Function("-", ConstantExpression::Integer(1), Column(decision));
			};
			outputs.push_back(
			    WithAlias(make_uniq<ComparisonExpression>(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
			                                              Column(decision), ConstantExpression::Number("0.5")),
			              question.name));
			outputs.push_back(WithAlias(Function("greatest", Column(decision), p_no()), question.name + "_confidence"));
			// NULL, not {yes: NULL, no: NULL}, when there is no answer
			auto probabilities = Function(
			    "map", Function("list_value", ConstantExpression::String("yes"), ConstantExpression::String("no")),
			    Function("list_value", Column(decision), p_no()));
			outputs.push_back(WithAlias(
			    Function("if", make_uniq<OperatorExpression>(ExpressionType::OPERATOR_IS_NULL, Column(decision)),
			             make_uniq<CastExpression>(probabilities_type, ConstantExpression::Null()),
			             std::move(probabilities)),
			    question.name + "_probabilities"));
		} else {
			outputs.push_back(WithAlias(
			    Function("struct_extract", Column(decision), ConstantExpression::String("choice")), question.name));
			for (auto field : {"confidence", "probabilities"}) {
				outputs.push_back(
				    WithAlias(Function("struct_extract", Column(decision), ConstantExpression::String(field)),
				              question.name + "_" + field));
			}
		}
	}
	inner->from_table = std::move(source);

	outer->select_list.push_back(std::move(star));
	for (auto &output : outputs) {
		outer->select_list.push_back(std::move(output));
	}
	outer->from_table = make_uniq<SubqueryRef>(MakeStatement(std::move(inner)));
	return outer;
}

//===--------------------------------------------------------------------===//
// Transforms
//===--------------------------------------------------------------------===//
// DecideTableRef <- DecideSource DecideKeyword? Parens(DecideBody) TableAlias?
static unique_ptr<TableRef> TransformDecideTableRef(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	auto source = transformer.Transform<unique_ptr<TableRef>>(SelectedChoice(list_pr.GetChild(0)));
	if (!list_pr.Child<OptionalParseResult>(1).HasResult()) {
		// `FROM t DECIDE (...)`: the keyword was read as the alias of t
		if (!StringUtil::CIEquals(source->alias.GetIdentifierName(), "decide") || !source->column_name_alias.empty()) {
			throw ParserException("syntax error at or near \"(\": expected DECIDE (...) after %s", source->ToString());
		}
		source->alias = Identifier();
	}
	auto spec = TransformDecideBody(transformer, PEGTransformerFactory::ExtractResultFromParens(list_pr.GetChild(2)));
	TableAlias alias;
	transformer.TransformOptional<TableAlias>(list_pr, 3, alias);

	auto result = make_uniq<SubqueryRef>(MakeStatement(LowerDecide(spec, std::move(source))), alias.name);
	result->column_name_alias = std::move(alias.column_name_alias);
	return std::move(result);
}

// DecideChooseCall <- 'DECISION_CHOOSE' Parens(DecideChooseCallBody)
// DecideChooseCallBody <- Expression DecideAmong DecideGiven
static unique_ptr<ParsedExpression> TransformChooseCall(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &body = PEGTransformerFactory::ExtractResultFromParens(parse_result.Cast<ListParseResult>().GetChild(1))
	                 .Cast<ListParseResult>();
	DecideQuestionSpec question;
	question.question = TransformExpression(transformer, body.GetChild(0));
	question.options = TransformAmong(transformer, body.GetChild(1));
	return ChooseCall(StateExpression(TransformGiven(transformer, body.GetChild(2))), question);
}

// DecideYesNoCall <- 'DECISION_YES_NO' Parens(DecideYesNoCallBody)
// DecideYesNoCallBody <- Expression DecideGiven
static unique_ptr<ParsedExpression> TransformYesNoCall(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &body = PEGTransformerFactory::ExtractResultFromParens(parse_result.Cast<ListParseResult>().GetChild(1))
	                 .Cast<ListParseResult>();
	DecideQuestionSpec question;
	question.question = TransformExpression(transformer, body.GetChild(0));
	return YesNoCall(StateExpression(TransformGiven(transformer, body.GetChild(1))), question);
}

//===--------------------------------------------------------------------===//
// Grammar extension
//===--------------------------------------------------------------------===//
static bool IsFunctionExpressionReference(const PEGExpression &expression) {
	return expression.type == PEGExpression::Type::REFERENCE &&
	       StringUtil::CIEquals(expression.text.GetString(), "FunctionExpression");
}

class DecideGrammarExtension final : public GrammarExtension {
public:
	static constexpr const char *NAME = "decide_onnx";

	DecideGrammarExtension()
	    : GrammarExtension(NAME, "DECIDE clause, decision_choose(question AMONG (...) GIVEN ...) and "
	                             "decision_yes_no(question GIVEN ...)") {
	}

	vector<GrammarChange> GetChanges() const override {
		vector<GrammarChange> changes;
		for (auto rule : DECIDE_RULES) {
			changes.push_back(GrammarChange::AddRule(rule));
		}
		changes.push_back(GrammarChange::SetTransformProcess(
		    "DecideTableRef", MakeTransform<unique_ptr<TableRef>>(TransformDecideTableRef)));
		changes.push_back(GrammarChange::SetTransformProcess(
		    "DecideChooseCall", MakeTransform<unique_ptr<ParsedExpression>>(TransformChooseCall)));
		changes.push_back(GrammarChange::SetTransformProcess(
		    "DecideYesNoCall", MakeTransform<unique_ptr<ParsedExpression>>(TransformYesNoCall)));
		changes.push_back(GrammarChange::PrependChoice("InnerTableRef", "DecideTableRef"));
		changes.push_back(
		    GrammarChange::PrependChoice("SingleExpression", "DecideChooseCall", IsFunctionExpressionReference));
		changes.push_back(
		    GrammarChange::PrependChoice("SingleExpression", "DecideYesNoCall", IsFunctionExpressionReference));
		return changes;
	}

	//! DuckDB's default grammar extended with only this extension (compiled once, shared by connections).
	shared_ptr<CompiledGrammar> GetCompiledGrammar() {
		lock_guard<mutex> guard(grammar_lock);
		if (!compiled_grammar) {
			vector<reference<GrammarExtension>> extensions;
			extensions.emplace_back(*this);
			compiled_grammar = CompiledGrammar::Create(extensions);
		}
		return compiled_grammar;
	}

	//! Adds this extension to the connection's active grammar extensions.
	void Activate(ClientContext &context) {
		auto &config = ClientConfig::GetConfig(context);
		auto &active = config.active_grammar_extensions;
		for (auto &name : active) {
			if (StringUtil::CIEquals(name, NAME)) {
				return;
			}
		}
		active.emplace_back(NAME);
		// Same effect as appending to `SET active_grammar_extensions`, without recompiling the common case.
		config.cached_grammar = active.size() == 1 ? GetCompiledGrammar() : CompiledGrammar::Create(context, active);
	}

private:
	mutex grammar_lock;
	shared_ptr<CompiledGrammar> compiled_grammar;
};

//! Connections opened after LOAD parse with the grammar from the start.
class DecideGrammarActivation final : public ExtensionCallback {
public:
	explicit DecideGrammarActivation(shared_ptr<DecideGrammarExtension> grammar_p) : grammar(std::move(grammar_p)) {
	}

	void OnConnectionOpened(ClientContext &context) override {
		grammar->Activate(context);
	}

private:
	shared_ptr<DecideGrammarExtension> grammar;
};

//! Connections already open at LOAD (including the one running it) switch on their own thread once their current or
//! next query ends, since another thread may be parsing with their grammar right now.
class PendingDecideGrammarActivation final : public ClientContextState {
public:
	static constexpr const char *KEY = "decide_onnx_pending_grammar_activation";

	explicit PendingDecideGrammarActivation(shared_ptr<DecideGrammarExtension> grammar_p)
	    : grammar(std::move(grammar_p)) {
	}

	void QueryEnd(ClientContext &context) override {
		grammar->Activate(context);
		// One-shot: a later `SET active_grammar_extensions` must stick.
		context.registered_state->Remove(KEY);
	}

private:
	shared_ptr<DecideGrammarExtension> grammar;
};

void RegisterDecideGrammar(DatabaseInstance &db) {
	auto grammar = make_shared_ptr<DecideGrammarExtension>();
	GrammarExtension::Register(db, grammar);
	ExtensionCallback::Register(DBConfig::GetConfig(db), make_shared_ptr<DecideGrammarActivation>(grammar));
	for (auto &connection : db.GetConnectionManager().GetConnectionList()) {
		connection->registered_state->GetOrCreate<PendingDecideGrammarActivation>(PendingDecideGrammarActivation::KEY,
		                                                                          grammar);
	}
}

} // namespace decide_onnx
} // namespace duckdb
