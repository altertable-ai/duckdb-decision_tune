#include "decide_onnx/model_store.hpp"

#include "decide_onnx/model_manifest.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "mbedtls_wrapper.hpp"

#include <random>

namespace duckdb {
namespace decide_onnx {

//! Where scripts/export uploads the files of ASSET_VERSION.
static constexpr const char *DEFAULT_MODEL_URL = "https://assets.altertable.ai/decide_onnx/v4";
static constexpr const char *DEFAULT_CACHE_DIR = "~/.duckdb/decide_onnx/v4";
//! Next to a verified file: "<sha256> <bytes> <mtime>", so later processes skip hashing the model again.
static constexpr const char *VERIFIED_SUFFIX = ".verified";
static constexpr idx_t IO_CHUNK = 8 * 1024 * 1024;

static string GetStringSetting(ClientContext &context, const string &name) {
	Value value;
	if (!context.TryGetCurrentSetting(Identifier(name), value) || value.IsNull()) {
		return string();
	}
	return value.ToString();
}

ModelStore &ModelStore::Instance() {
	// Never destroyed: ONNX Runtime sessions must not outlive their Ort::Env, and tearing either down during process
	// or DLL unload is unsafe, so both live until the process exits.
	static auto *store = new ModelStore();
	return *store;
}

static string CacheDir(ClientContext &context) {
	auto dir = GetStringSetting(context, "decide_onnx_cache_dir");
	if (dir.empty()) {
		dir = DEFAULT_CACHE_DIR;
	}
	return FileSystem::GetFileSystem(context).ExpandPath(dir);
}

string ModelStore::CachePath(ClientContext &context, const ModelFile &file) {
	return FileSystem::GetFileSystem(context).JoinPath(CacheDir(context), file.file_name);
}

string ModelStore::FileUrl(ClientContext &context, const ModelFile &file) {
	auto base = GetStringSetting(context, "decide_onnx_model_url");
	if (base.empty()) {
		base = DEFAULT_MODEL_URL;
	}
	while (!base.empty() && base.back() == '/') {
		base.pop_back();
	}
	return base + "/" + file.file_name;
}

bool ModelStore::IsCached(ClientContext &context, const ModelFile &file) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto path = CachePath(context, file);
	if (!fs.FileExists(path)) {
		return false;
	}
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	return NumericCast<uint64_t>(handle->GetFileSize()) == file.file_bytes;
}

//===--------------------------------------------------------------------===//
// Verification
//===--------------------------------------------------------------------===//
static string Sha256Hex(duckdb_mbedtls::MbedTlsWrapper::SHA256State &state) {
	char hex[64];
	state.FinishHex(hex);
	return string(hex, 64);
}

static string FileStamp(FileSystem &fs, const string &path, const ModelFile &file) {
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	return StringUtil::Format("%s %llu %lld", file.sha256, NumericCast<uint64_t>(handle->GetFileSize()),
	                          fs.GetLastModifiedTime(*handle).value);
}

static string ReadSmallFile(FileSystem &fs, const string &path) {
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto size = NumericCast<idx_t>(handle->GetFileSize());
	string result(size, '\0');
	handle->Read(&result[0], size, 0);
	return result;
}

static void WriteVerifiedStamp(FileSystem &fs, const string &path, const ModelFile &file) {
	auto stamp = FileStamp(fs, path, file);
	auto marker = path + VERIFIED_SUFFIX;
	try {
		auto handle = fs.OpenFile(marker, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
		handle->Write(const_cast<char *>(stamp.data()), stamp.size());
		handle->Sync();
	} catch (std::exception &) {
		// Only an optimization: without the marker the next process hashes the file again.
		fs.TryRemoveFile(marker);
	}
}

static bool HasVerifiedStamp(FileSystem &fs, const string &path, const ModelFile &file) {
	auto marker = path + VERIFIED_SUFFIX;
	if (!fs.FileExists(marker)) {
		return false;
	}
	try {
		return ReadSmallFile(fs, marker) == FileStamp(fs, path, file);
	} catch (std::exception &) {
		return false;
	}
}

static void VerifyCachedFile(FileSystem &fs, const string &path, const ModelFile &file) {
	if (HasVerifiedStamp(fs, path, file)) {
		return;
	}
	duckdb_mbedtls::MbedTlsWrapper::SHA256State sha;
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto buffer = make_unsafe_uniq_array<data_t>(IO_CHUNK);
	while (true) {
		auto read = handle->Read(buffer.get(), IO_CHUNK);
		if (read <= 0) {
			break;
		}
		sha.AddBytes(buffer.get(), NumericCast<idx_t>(read));
	}
	handle.reset();
	auto actual = Sha256Hex(sha);
	if (actual != file.sha256) {
		fs.TryRemoveFile(path);
		fs.TryRemoveFile(path + VERIFIED_SUFFIX);
		throw IOException("decide_onnx: checksum mismatch for %s (expected %s, got %s); the corrupted file '%s' was "
		                  "removed, run the query again to download it",
		                  file.file_name, file.sha256, actual, path);
	}
	fs.TryRemoveFile(path + VERIFIED_SUFFIX);
	WriteVerifiedStamp(fs, path, file);
}

//===--------------------------------------------------------------------===//
// Download
//===--------------------------------------------------------------------===//
static void Download(ClientContext &context, const ModelFile &file, const string &target) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto url = ModelStore::FileUrl(context, file);
	if (StringUtil::StartsWith(url, "http://") || StringUtil::StartsWith(url, "https://")) {
		ExtensionHelper::TryAutoLoadExtension(context, "httpfs");
	}
	fs.CreateDirectoriesRecursive(CacheDir(context));
	// A name unique to this download, so concurrent downloads (other processes, other databases) never write to
	// the same file; the final rename is atomic.
	std::random_device random;
	auto tmp = StringUtil::Format("%s.tmp.%08x%08x", target, random(), random());
	duckdb_mbedtls::MbedTlsWrapper::SHA256State sha;
	try {
		auto source = fs.OpenFile(url, FileFlags::FILE_FLAGS_READ);
		auto sink = fs.OpenFile(tmp, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
		auto buffer = make_unsafe_uniq_array<data_t>(IO_CHUNK);
		while (true) {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			auto read = source->Read(buffer.get(), IO_CHUNK);
			if (read <= 0) {
				break;
			}
			sha.AddBytes(buffer.get(), NumericCast<idx_t>(read));
			sink->Write(buffer.get(), NumericCast<idx_t>(read));
		}
		sink->Sync();
	} catch (std::exception &ex) {
		fs.TryRemoveFile(tmp);
		ErrorData error(ex);
		throw IOException("decide_onnx: could not download %s: %s. To use local files instead, SET "
		                  "decide_onnx_model_path = '/path/to/vela3-fast-onnx'",
		                  url, error.RawMessage());
	}
	auto actual = Sha256Hex(sha);
	if (actual != file.sha256) {
		fs.TryRemoveFile(tmp);
		throw IOException("decide_onnx: checksum mismatch for %s (expected %s, got %s) after downloading %s",
		                  file.file_name, file.sha256, actual, url);
	}
	fs.TryRemoveFile(target + VERIFIED_SUFFIX);
	fs.MoveFile(tmp, target);
	WriteVerifiedStamp(fs, target, file);
}

//===--------------------------------------------------------------------===//
// Model records
//===--------------------------------------------------------------------===//
ModelStore::Location ModelStore::Resolve(ClientContext &context) {
	auto &fs = FileSystem::GetFileSystem(context);
	Location location;
	auto explicit_path = GetStringSetting(context, "decide_onnx_model_path");
	location.explicit_path = !explicit_path.empty();
	location.directory = location.explicit_path ? fs.ExpandPath(explicit_path) : CacheDir(context);
	if (!fs.IsPathAbsolute(location.directory)) {
		location.directory = fs.JoinPath(FileSystem::GetWorkingDirectory(), location.directory);
	}
	if (location.explicit_path && !fs.DirectoryExists(location.directory)) {
		throw IOException("decide_onnx: decide_onnx_model_path '%s' is not a directory (it should hold "
		                  "tokenizer.json and model.onnx)",
		                  location.directory);
	}
	return location;
}

shared_ptr<ModelStore::ModelRecord> ModelStore::Record(const string &directory) {
	lock_guard<mutex> guard(lock);
	auto &record = models[directory];
	if (!record) {
		record = make_shared_ptr<ModelRecord>();
	}
	return record;
}

string ModelStore::EnsureFile(ClientContext &context, const Location &location, const ModelFile &file, bool &ready) {
	auto &fs = FileSystem::GetFileSystem(context);
	if (location.explicit_path) {
		// Not checksummed.
		auto path = fs.JoinPath(location.directory, file.file_name);
		if (fs.FileExists(path)) {
			return path;
		}
		throw IOException("decide_onnx: decide_onnx_model_path '%s' has no %s", location.directory, file.file_name);
	}
	auto path = fs.JoinPath(location.directory, file.file_name);
	if (!ready) {
		if (!IsCached(context, file)) {
			Download(context, file, path);
		} else {
			VerifyCachedFile(fs, path, file);
		}
		ready = true;
	}
	return path;
}

shared_ptr<BpeTokenizer> ModelStore::LoadTokenizer(ClientContext &context, const Location &location,
                                                   ModelRecord &record) {
	auto path = EnsureFile(context, location, TOKENIZER_FILE, record.tokenizer_ready);
	if (!record.tokenizer) {
		auto &fs = FileSystem::GetFileSystem(context);
		record.tokenizer = make_shared_ptr<BpeTokenizer>(ReadSmallFile(fs, path), path);
	}
	return record.tokenizer;
}

shared_ptr<BpeTokenizer> ModelStore::GetTokenizer(ClientContext &context) {
	auto location = Resolve(context);
	auto record = Record(location.directory);
	lock_guard<mutex> guard(record->lock);
	return LoadTokenizer(context, location, *record);
}

shared_ptr<DecisionModel> ModelStore::GetModel(ClientContext &context) {
	auto location = Resolve(context);
	auto record = Record(location.directory);
	lock_guard<mutex> guard(record->lock);
	if (!record->model) {
		auto tokenizer = LoadTokenizer(context, location, *record);
		auto onnx_path = EnsureFile(context, location, ONNX_FILE, record->onnx_ready);
		record->model = make_shared_ptr<DecisionModel>(tokenizer, make_shared_ptr<OrtScorer>(onnx_path));
	}
	return record->model;
}

//===--------------------------------------------------------------------===//
// decide_onnx_models()
//===--------------------------------------------------------------------===//
struct ModelsGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
};

static unique_ptr<FunctionData> ModelsBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<Identifier> &names) {
	names = {"model", "file_name", "url", "sha256", "file_bytes", "cached", "local_path"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::UBIGINT, LogicalType::BOOLEAN, LogicalType::VARCHAR};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> ModelsInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<ModelsGlobalState>();
}

static void ModelsFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<ModelsGlobalState>();
	constexpr idx_t total = sizeof(MODEL_FILES) / sizeof(MODEL_FILES[0]);
	while (state.offset < total) {
		auto &file = *MODEL_FILES[state.offset++];
		output.data[0].Append(Value(MODEL_NAME));
		output.data[1].Append(Value(file.file_name));
		output.data[2].Append(Value(ModelStore::FileUrl(context, file)));
		output.data[3].Append(Value(file.sha256));
		output.data[4].Append(Value::UBIGINT(file.file_bytes));
		output.data[5].Append(Value::BOOLEAN(ModelStore::IsCached(context, file)));
		output.data[6].Append(Value(ModelStore::CachePath(context, file)));
	}
}

void RegisterModelStore(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption("decide_onnx_model_url",
	                          "Base URL the vela3-fast tokenizer.json and model.onnx are downloaded from",
	                          LogicalType::VARCHAR, Value(DEFAULT_MODEL_URL));
	config.AddExtensionOption("decide_onnx_cache_dir", "Directory downloaded vela3-fast files are cached in",
	                          LogicalType::VARCHAR, Value(DEFAULT_CACHE_DIR));
	config.AddExtensionOption("decide_onnx_model_path",
	                          "Local directory with tokenizer.json and model.onnx to use instead of the download cache "
	                          "(no checksum check)",
	                          LogicalType::VARCHAR, Value(""));

	TableFunction models("decide_onnx_models", {}, ModelsFunction, ModelsBind, ModelsInit);
	loader.RegisterFunction(models);
}

} // namespace decide_onnx
} // namespace duckdb
