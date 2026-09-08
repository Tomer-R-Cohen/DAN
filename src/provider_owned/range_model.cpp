#include "provider_owned/range_model.hpp"

#include "platform.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dan::provider_owned {
namespace {

namespace fs = std::filesystem;

struct Tensor {
    std::string name;
    std::uint64_t offset = 0;
};

struct Index {
    std::string architecture;
    std::uint32_t layers = 0;
    std::uint32_t hidden = 0;
    std::uint32_t heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint64_t logical_size = 0;
    std::uint64_t data_offset = 0;
    std::uint32_t alignment = 32;
    std::vector<Tensor> tensors;
};

struct Range {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::string sha256;
};

constexpr std::uint64_t metadata_chunk = 1024 * 1024;
constexpr std::uint64_t max_metadata = 64 * 1024 * 1024;

std::string lowercase(std::string value) {
    for (char& byte : value) byte = static_cast<char>(
        std::tolower(static_cast<unsigned char>(byte)));
    return value;
}

bool parse_u64(std::string_view text, std::uint64_t& value) {
    const auto [end, status] = std::from_chars(text.data(), text.data() + text.size(), value);
    return status == std::errc{} && end == text.data() + text.size();
}

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    template <typename T>
    bool integer(T& value) {
        if (cursor_ > bytes_.size() || bytes_.size() - cursor_ < sizeof(T)) return false;
        value = 0;
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            value |= static_cast<T>(bytes_[cursor_ + i]) << (i * 8);
        }
        cursor_ += sizeof(T);
        return true;
    }

    bool skip(std::uint64_t count) {
        if (count > bytes_.size() || cursor_ > bytes_.size() - count) return false;
        cursor_ += static_cast<std::size_t>(count);
        return true;
    }

    bool string(std::string& value) {
        std::uint64_t size = 0;
        if (!integer(size) || size > 1024 * 1024 * 32 || !available(size)) return false;
        value.assign(reinterpret_cast<const char*>(bytes_.data() + cursor_),
            static_cast<std::size_t>(size));
        cursor_ += static_cast<std::size_t>(size);
        return true;
    }

    bool available(std::uint64_t count) const {
        return count <= bytes_.size() && cursor_ <= bytes_.size() - count;
    }

    std::size_t cursor() const { return cursor_; }

private:
    std::span<const std::uint8_t> bytes_;
    std::size_t cursor_ = 0;
};

bool skip_value(Reader& reader, std::uint32_t type, int depth = 0) {
    if (depth > 4) return false;
    switch (type) {
    case 0: case 1: case 7: return reader.skip(1);
    case 2: case 3: return reader.skip(2);
    case 4: case 5: case 6: return reader.skip(4);
    case 10: case 11: case 12: return reader.skip(8);
    case 8: {
        std::string ignored;
        return reader.string(ignored);
    }
    case 9: {
        std::uint32_t element_type = 0;
        std::uint64_t count = 0;
        if (!reader.integer(element_type) || element_type > 12 || !reader.integer(count)
            || count > 100000000) return false;
        for (std::uint64_t i = 0; i < count; ++i) {
            if (!skip_value(reader, element_type, depth + 1)) return false;
        }
        return true;
    }
    default: return false;
    }
}

enum class ParseResult { complete, incomplete, invalid };

ParseResult parse_index(std::span<const std::uint8_t> bytes,
    std::uint64_t logical_size, Index& index, std::string& error) {
    Reader reader(bytes);
    std::uint32_t magic = 0, version = 0;
    std::uint64_t tensor_count = 0, metadata_count = 0;
    if (!reader.integer(magic) || !reader.integer(version)
        || !reader.integer(tensor_count) || !reader.integer(metadata_count)) {
        return ParseResult::incomplete;
    }
    if (magic != 0x46554747 || version < 2 || version > 3
        || tensor_count == 0 || tensor_count > 1000000 || metadata_count > 1000000) {
        error = "invalid GGUF header";
        return ParseResult::invalid;
    }
    std::uint32_t alignment = 32;
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        std::string key;
        std::uint32_t type = 0;
        if (!reader.string(key) || !reader.integer(type)) return ParseResult::incomplete;
        if (type > 12) { error = "invalid GGUF metadata type"; return ParseResult::invalid; }
        if (key == "general.architecture" && type == 8) {
            if (!reader.string(index.architecture)) return ParseResult::incomplete;
        } else if ((key == "qwen2.block_count" || key == "qwen2.embedding_length"
                || key == "qwen2.attention.head_count"
                || key == "qwen2.attention.head_count_kv") && type == 4) {
            std::uint32_t value = 0;
            if (!reader.integer(value)) return ParseResult::incomplete;
            if (key == "qwen2.block_count") index.layers = value;
            else if (key == "qwen2.embedding_length") index.hidden = value;
            else if (key == "qwen2.attention.head_count") index.heads = value;
            else index.kv_heads = value;
        } else if (key == "general.alignment" && type == 4) {
            if (!reader.integer(alignment)) return ParseResult::incomplete;
        } else if (!skip_value(reader, type)) return ParseResult::incomplete;
    }
    if (alignment == 0 || alignment > 1024 * 1024
        || (alignment & (alignment - 1)) != 0) {
        error = "invalid GGUF alignment";
        return ParseResult::invalid;
    }
    std::vector<Tensor> tensors;
    tensors.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        Tensor tensor;
        std::uint32_t dimensions = 0, type = 0;
        if (!reader.string(tensor.name) || tensor.name.empty()
            || !reader.integer(dimensions)) return ParseResult::incomplete;
        if (dimensions == 0 || dimensions > 4) {
            error = "invalid GGUF tensor dimensions";
            return ParseResult::invalid;
        }
        for (std::uint32_t dimension = 0; dimension < dimensions; ++dimension) {
            std::uint64_t ignored = 0;
            if (!reader.integer(ignored)) return ParseResult::incomplete;
        }
        if (!reader.integer(type) || !reader.integer(tensor.offset)) {
            return ParseResult::incomplete;
        }
        (void) type;
        tensors.push_back(std::move(tensor));
    }
    const std::uint64_t metadata_end = reader.cursor();
    const std::uint64_t data_offset = (metadata_end + alignment - 1)
        & ~(static_cast<std::uint64_t>(alignment) - 1);
    if (data_offset >= logical_size) {
        error = "GGUF data offset exceeds file size";
        return ParseResult::invalid;
    }
    std::sort(tensors.begin(), tensors.end(), [](const Tensor& a, const Tensor& b) {
        return a.offset < b.offset;
    });
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].offset >= logical_size - data_offset
            || (i != 0 && tensors[i - 1].offset >= tensors[i].offset)) {
            error = "invalid or overlapping GGUF tensor offsets";
            return ParseResult::invalid;
        }
    }
    index.logical_size = logical_size;
    index.data_offset = data_offset;
    index.alignment = alignment;
    index.tensors = std::move(tensors);
    return ParseResult::complete;
}

std::optional<int> block_layer(std::string_view name) {
    if (!name.starts_with("blk.")) return std::nullopt;
    const std::size_t end = name.find('.', 4);
    if (end == std::string_view::npos) return std::nullopt;
    int layer = -1;
    const auto [parsed, status] = std::from_chars(name.data() + 4, name.data() + end, layer);
    if (status != std::errc{} || parsed != name.data() + end) return std::nullopt;
    return layer;
}

bool owned_tensor(const std::string& name, int begin, int end, int layers,
    bool output_present) {
    if (const auto layer = block_layer(name)) return *layer >= begin && *layer < end;
    if (name == "token_embd.weight") {
        return begin == 0 || (end == layers && !output_present);
    }
    if (name.starts_with("output_norm.") || name.starts_with("output.")) {
        return end == layers;
    }
    return false;
}

bool read_file(const fs::path& path, std::vector<std::uint8_t>& bytes, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "could not open downloaded range"; return false; }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > max_metadata) {
        error = "downloaded metadata range is too large"; return false;
    }
    input.seekg(0);
    const std::size_t old = bytes.size();
    bytes.resize(old + static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data() + old), size);
    if (!input) { error = "could not read downloaded range"; return false; }
    return true;
}

std::optional<std::string> header_value(const std::string& headers,
    std::string_view name) {
    const std::string lower = lowercase(headers);
    const std::string needle = lowercase(std::string(name)) + ':';
    std::size_t found = 0;
    std::optional<std::string> result;
    while ((found = lower.find(needle, found)) != std::string::npos) {
        if (found == 0 || lower[found - 1] == '\n') {
            std::size_t begin = found + needle.size();
            while (begin < headers.size() && (headers[begin] == ' ' || headers[begin] == '\t')) ++begin;
            std::size_t end = headers.find_first_of("\r\n", begin);
            std::string value = headers.substr(begin, end - begin);
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            result = value;
        }
        found += needle.size();
    }
    return result;
}

bool fetch_range(const RangeModelRequest& request, std::uint64_t offset,
    std::uint64_t size, const fs::path& output, std::uint64_t& logical_size,
    std::string& error, bool progress = false) {
    if (size == 0 || offset > std::numeric_limits<std::uint64_t>::max() - size) {
        error = "invalid HTTP range"; return false;
    }
    const fs::path headers = output.string() + ".headers";
    const std::string range = std::to_string(offset) + '-'
        + std::to_string(offset + size - 1);
    std::vector<std::string> arguments{"curl", "--fail", "--location", "--show-error"};
    if (!progress) arguments.push_back("--silent");
    arguments.insert(arguments.end(), {"--retry", "5", "--range", range,
        "--max-filesize", std::to_string(size), "--dump-header", headers.string(),
        "--output", output.string(), request.url});
    const bool downloaded = dan::platform::run(arguments, error);
    if (!downloaded) {
        fs::remove(output); fs::remove(headers);
        error = "range download failed: " + error;
        return false;
    }
    std::error_code ec;
    if (fs::file_size(output, ec) != size || ec) {
        fs::remove(output); fs::remove(headers);
        error = "server returned a truncated or non-range response";
        return false;
    }
    std::ifstream header_input(headers, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(header_input)), {});
    header_input.close();
    fs::remove(headers);
    const auto content_range = header_value(text, "content-range");
    if (!content_range) { fs::remove(output); error = "missing Content-Range"; return false; }
    const std::string expected = "bytes " + range + '/';
    if (!content_range->starts_with(expected)
        || !parse_u64(std::string_view(*content_range).substr(expected.size()), logical_size)) {
        fs::remove(output); error = "wrong Content-Range"; return false;
    }
    if (const auto revision = header_value(text, "x-repo-commit");
        !revision || lowercase(*revision) != lowercase(request.revision)) {
        fs::remove(output); error = "model revision mismatch"; return false;
    }
    if (const auto hash = header_value(text, "x-linked-etag");
        !hash || lowercase(*hash) != lowercase(request.full_sha256)) {
        fs::remove(output); error = "remote model SHA-256 mismatch"; return false;
    }
    return true;
}

bool sparse_create(const fs::path& path, std::uint64_t size, std::string& error) {
#ifdef _WIN32
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
        nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) { error = "could not create sparse model file"; return false; }
    DWORD returned = 0;
    LARGE_INTEGER end{}; end.QuadPart = static_cast<LONGLONG>(size);
    const bool ok = DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0,
            &returned, nullptr)
        && SetFilePointerEx(file, end, nullptr, FILE_BEGIN) && SetEndOfFile(file);
    CloseHandle(file);
#else
    const int file = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    const bool ok = file != -1 && ftruncate(file, static_cast<off_t>(size)) == 0;
    if (file != -1) close(file);
#endif
    if (!ok) { fs::remove(path); error = "could not initialize sparse model file"; }
    return ok;
}

std::uint64_t physical_size(const fs::path& path) {
#ifdef _WIN32
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(file);
        CloseHandle(file);
    }
    DWORD high = 0;
    const DWORD low = GetCompressedFileSizeW(path.c_str(), &high);
    if (low == INVALID_FILE_SIZE && GetLastError() != NO_ERROR) return 0;
    return (static_cast<std::uint64_t>(high) << 32) | low;
#else
    struct stat value{};
    return stat(path.c_str(), &value) == 0
        ? static_cast<std::uint64_t>(value.st_blocks) * 512 : 0;
#endif
}

bool copy_into(const fs::path& source, const fs::path& target,
    std::uint64_t offset, std::string& error) {
    std::ifstream input(source, std::ios::binary);
    std::fstream output(target, std::ios::binary | std::ios::in | std::ios::out);
    if (!input || !output) { error = "could not open sparse range files"; return false; }
    output.seekp(static_cast<std::streamoff>(offset));
    std::vector<char> buffer(1024 * 1024);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) output.write(buffer.data(), count);
    }
    if (input.bad() || !output) { error = "could not write sparse model range"; return false; }
    return true;
}

bool hash_range(const fs::path& model, const Range& range, const fs::path& temporary,
    std::string& digest, std::string& error) {
    std::ifstream input(model, std::ios::binary);
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!input || !output) { error = "could not open range for verification"; return false; }
    input.seekg(static_cast<std::streamoff>(range.offset));
    std::uint64_t remaining = range.size;
    std::vector<char> buffer(1024 * 1024);
    while (remaining != 0) {
        const auto count = static_cast<std::streamsize>(
            std::min<std::uint64_t>(remaining, buffer.size()));
        input.read(buffer.data(), count);
        if (input.gcount() != count) { error = "partial sparse range"; return false; }
        output.write(buffer.data(), count);
        if (!output) { error = "could not stage range verification"; return false; }
        remaining -= static_cast<std::uint64_t>(count);
    }
    output.close();
    const bool ok = dan::platform::sha256_file(temporary, digest, error);
    fs::remove(temporary);
    return ok;
}

std::vector<Range> required_ranges(const Index& index, int begin, int end,
    std::uint64_t& tensor_count, std::uint64_t& shared_bytes) {
    int layers = 0;
    bool output_present = false;
    for (const Tensor& tensor : index.tensors) {
        if (const auto layer = block_layer(tensor.name)) layers = std::max(layers, *layer + 1);
        if (tensor.name == "output.weight") output_present = true;
    }
    std::vector<Range> ranges{{0, index.data_offset, {}}};
    tensor_count = 0;
    shared_bytes = 0;
    for (std::size_t i = 0; i < index.tensors.size(); ++i) {
        const Tensor& tensor = index.tensors[i];
        if (!owned_tensor(tensor.name, begin, end, layers, output_present)) continue;
        const std::uint64_t start = index.data_offset + tensor.offset;
        const std::uint64_t finish = i + 1 == index.tensors.size()
            ? index.logical_size : index.data_offset + index.tensors[i + 1].offset;
        ++tensor_count;
        if (tensor.name == "token_embd.weight" && begin != 0) shared_bytes += finish - start;
        if (ranges.back().offset + ranges.back().size == start) ranges.back().size += finish - start;
        else ranges.push_back({start, finish - start, {}});
    }
    return ranges;
}

fs::path sidecar_path(const fs::path& model) { return model.string() + ".ranges"; }

bool write_sidecar(const RangeModelRequest& request, const RangeModelStats& stats,
    const std::vector<Range>& ranges, std::string& error) {
    const fs::path target = sidecar_path(request.path);
    const fs::path temporary = target.string() + ".partial";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) { error = "could not create range-cache metadata"; return false; }
    output << "DAN_RANGE_CACHE_V1\nurl=" << request.url
        << "\nrevision=" << lowercase(request.revision)
        << "\nfull_sha256=" << lowercase(request.full_sha256)
        << "\nstage_start=" << request.stage_start
        << "\nstage_end=" << request.stage_end
        << "\nlogical_bytes=" << stats.logical_bytes
        << "\nheader_bytes=" << stats.header_bytes
        << "\ndownloaded_bytes=" << stats.downloaded_bytes
        << "\nshared_bytes=" << stats.shared_bytes
        << "\ntensors_present=" << stats.tensors_present << '\n';
    for (const Range& range : ranges) {
        output << "range=" << range.offset << ',' << range.size << ',' << range.sha256 << '\n';
    }
    output.close();
    if (!output) { fs::remove(temporary); error = "could not write range-cache metadata"; return false; }
    std::error_code ec;
    fs::rename(temporary, target, ec);
    if (ec) { fs::remove(temporary); error = "could not publish range-cache metadata"; return false; }
    return true;
}

bool read_sidecar(const RangeModelRequest& request, RangeModelStats& stats,
    std::vector<Range>& ranges, std::string& error) {
    std::ifstream input(sidecar_path(request.path), std::ios::binary);
    if (!input) return false;
    std::string line;
    if (!std::getline(input, line) || line != "DAN_RANGE_CACHE_V1") {
        error = "invalid range-cache metadata"; return false;
    }
    std::string url, revision, hash;
    int begin = -1, end = -1;
    while (std::getline(input, line)) {
        const std::size_t equal = line.find('=');
        if (equal == std::string::npos) { error = "invalid range-cache metadata"; return false; }
        const std::string key = line.substr(0, equal);
        const std::string value = line.substr(equal + 1);
        std::uint64_t number = 0;
        if (key == "url") url = value;
        else if (key == "revision") revision = value;
        else if (key == "full_sha256") hash = value;
        else if (key == "stage_start") {
            std::uint64_t parsed = 0;
            if (!parse_u64(value, parsed) || parsed > std::numeric_limits<int>::max()) {
                error = "invalid cached stage"; return false;
            }
            begin = static_cast<int>(parsed);
        } else if (key == "stage_end") {
            std::uint64_t parsed = 0;
            if (!parse_u64(value, parsed) || parsed > std::numeric_limits<int>::max()) {
                error = "invalid cached stage"; return false;
            }
            end = static_cast<int>(parsed);
        }
        else if (key == "logical_bytes" && parse_u64(value, number)) stats.logical_bytes = number;
        else if (key == "header_bytes" && parse_u64(value, number)) stats.header_bytes = number;
        else if (key == "downloaded_bytes" && parse_u64(value, number)) stats.downloaded_bytes = number;
        else if (key == "shared_bytes" && parse_u64(value, number)) stats.shared_bytes = number;
        else if (key == "tensors_present" && parse_u64(value, number)) stats.tensors_present = number;
        else if (key == "range") {
            const std::size_t first = value.find(','), second = value.find(',', first + 1);
            Range range;
            if (first == std::string::npos || second == std::string::npos
                || !parse_u64(std::string_view(value).substr(0, first), range.offset)
                || !parse_u64(std::string_view(value).substr(first + 1, second - first - 1), range.size)) {
                error = "invalid cached range"; return false;
            }
            range.sha256 = value.substr(second + 1);
            ranges.push_back(std::move(range));
        } else { error = "invalid range-cache metadata"; return false; }
    }
    if (url != request.url || revision != lowercase(request.revision)
        || hash != lowercase(request.full_sha256) || begin != request.stage_start
        || end != request.stage_end || ranges.empty()) {
        error = "range cache does not match model revision or stage";
        return false;
    }
    return true;
}

bool reuse_cache(const RangeModelRequest& request, RangeModelStats& stats,
    std::string& error) {
    std::vector<Range> ranges;
    if (!read_sidecar(request, stats, ranges, error)) return false;
    std::error_code ec;
    if (!fs::is_regular_file(request.path, ec) || ec
        || fs::file_size(request.path, ec) != stats.logical_bytes || ec) {
        error = "partial model cache is missing or truncated"; return false;
    }
    const fs::path verify = request.path.string() + ".verify";
    for (const Range& range : ranges) {
        std::string digest;
        if (range.offset > stats.logical_bytes || range.size > stats.logical_bytes - range.offset
            || !hash_range(request.path, range, verify, digest, error)
            || lowercase(digest) != lowercase(range.sha256)) {
            fs::remove(verify);
            if (error.empty()) error = "cached range SHA-256 mismatch";
            return false;
        }
    }
    stats.physical_bytes = physical_size(request.path);
    stats.cache_reused = true;
    return true;
}

} // namespace

bool inspect_range_model(const RangeModelRequest& request, ModelIndex& output,
    std::string& error) {
    output = {};
    if (request.url.empty() || request.path.empty()) {
        error = "model inspection requires a URL and temporary path";
        return false;
    }
    std::vector<std::uint8_t> metadata;
    std::uint64_t logical_size = 0;
    Index parsed_index;
    const fs::path temporary = request.path.string() + ".inspect";
    ParseResult parsed = ParseResult::incomplete;
    for (std::uint64_t offset = 0; offset < max_metadata && parsed == ParseResult::incomplete;
        offset += metadata_chunk) {
        const std::uint64_t wanted = logical_size == 0 ? metadata_chunk
            : std::min(metadata_chunk, logical_size - offset);
        std::uint64_t reported = 0;
        if (wanted == 0 || !fetch_range(request, offset, wanted, temporary, reported, error)) {
            fs::remove(temporary);
            return false;
        }
        if (logical_size != 0 && logical_size != reported) {
            fs::remove(temporary); error = "remote model size changed"; return false;
        }
        logical_size = reported;
        if (!read_file(temporary, metadata, error)) { fs::remove(temporary); return false; }
        fs::remove(temporary);
        parsed = parse_index(metadata, logical_size, parsed_index, error);
    }
    fs::remove(temporary);
    if (parsed != ParseResult::complete || parsed_index.architecture != "qwen2"
        || parsed_index.layers < 2 || parsed_index.hidden == 0
        || parsed_index.heads == 0 || parsed_index.kv_heads == 0) {
        if (error.empty()) error = "unsupported or incomplete Qwen2 GGUF metadata";
        return false;
    }
    output.architecture = parsed_index.architecture;
    output.layers = parsed_index.layers;
    output.hidden = parsed_index.hidden;
    output.heads = parsed_index.heads;
    output.kv_heads = parsed_index.kv_heads;
    output.logical_bytes = parsed_index.logical_size;
    output.header_bytes = parsed_index.data_offset;
    output.tensors.reserve(parsed_index.tensors.size());
    for (std::size_t i = 0; i < parsed_index.tensors.size(); ++i) {
        const std::uint64_t finish = i + 1 == parsed_index.tensors.size()
            ? parsed_index.logical_size - parsed_index.data_offset
            : parsed_index.tensors[i + 1].offset;
        output.tensors.push_back({parsed_index.tensors[i].name,
            finish - parsed_index.tensors[i].offset});
    }
    return true;
}

std::uint64_t stage_model_bytes(const ModelIndex& index, int begin, int end) {
    if (begin < 0 || end <= begin || end > static_cast<int>(index.layers)) return 0;
    bool output_present = false;
    for (const ModelTensor& tensor : index.tensors) {
        if (tensor.name == "output.weight") output_present = true;
    }
    std::uint64_t bytes = index.header_bytes;
    for (const ModelTensor& tensor : index.tensors) {
        if (owned_tensor(tensor.name, begin, end, static_cast<int>(index.layers),
                output_present)) {
            if (bytes > std::numeric_limits<std::uint64_t>::max() - tensor.bytes) return 0;
            bytes += tensor.bytes;
        }
    }
    return bytes;
}

bool prepare_range_model(const RangeModelRequest& request, RangeModelStats& stats,
    std::string& error) {
    stats = {};
    const auto hex = [](std::string_view value, std::size_t size) {
        return value.size() == size && std::ranges::all_of(value, [](unsigned char byte) {
            return std::isxdigit(byte) != 0;
        });
    };
    if (request.url.empty() || request.revision.size() != 40
        || !hex(request.revision, 40) || !hex(request.full_sha256, 64) || request.path.empty()
        || request.stage_start < 0 || request.stage_end <= request.stage_start) {
        error = "invalid range-model request"; return false;
    }
    if (fs::exists(request.path) || fs::exists(sidecar_path(request.path))) {
        if (reuse_cache(request, stats, error)) return true;
        std::ifstream marker(sidecar_path(request.path));
        std::string first;
        const bool ours = std::getline(marker, first) && first == "DAN_RANGE_CACHE_V1";
        marker.close();
        if (!ours) {
            return false; // Never overwrite or silently treat a full file as a range cache.
        }
        fs::remove(request.path);
        fs::remove(sidecar_path(request.path));
        error.clear();
    }
    std::error_code ec;
    if (!request.path.parent_path().empty()) fs::create_directories(request.path.parent_path(), ec);
    if (ec) { error = "could not create range-cache directory"; return false; }

    std::vector<std::uint8_t> metadata;
    std::uint64_t logical_size = 0;
    std::uint64_t transferred = 0;
    Index index;
    const fs::path temporary = request.path.string() + ".download";
    ParseResult parsed = ParseResult::incomplete;
    for (std::uint64_t offset = 0; offset < max_metadata && parsed == ParseResult::incomplete;
        offset += metadata_chunk) {
        const std::uint64_t wanted = logical_size == 0
            ? metadata_chunk : std::min(metadata_chunk, logical_size - offset);
        std::uint64_t reported_size = 0;
        if (wanted == 0 || !fetch_range(request, offset, wanted, temporary,
                reported_size, error)) return false;
        if (logical_size != 0 && reported_size != logical_size) {
            fs::remove(temporary); error = "remote model size changed"; return false;
        }
        logical_size = reported_size;
        transferred += wanted;
        if (!read_file(temporary, metadata, error)) { fs::remove(temporary); return false; }
        fs::remove(temporary);
        parsed = parse_index(metadata, logical_size, index, error);
    }
    if (parsed != ParseResult::complete) {
        if (error.empty()) error = "GGUF metadata exceeds range parser limit";
        return false;
    }
    int layers = 0;
    for (const Tensor& tensor : index.tensors) {
        if (const auto layer = block_layer(tensor.name)) layers = std::max(layers, *layer + 1);
    }
    if (request.stage_end > layers) { error = "stage exceeds GGUF layer count"; return false; }

    std::uint64_t tensors = 0, shared = 0;
    std::vector<Range> ranges = required_ranges(index, request.stage_start,
        request.stage_end, tensors, shared);
    if (tensors == 0) { error = "stage owns no GGUF tensors"; return false; }

    std::uint64_t reported_size = 0;
    if (!sparse_create(request.path, logical_size, error)) { fs::remove(temporary); return false; }

    stats.logical_bytes = logical_size;
    stats.header_bytes = index.data_offset;
    stats.tensors_present = tensors;
    stats.shared_bytes = shared;
    stats.downloaded_bytes = transferred;
    for (Range& range : ranges) {
        if (range.offset == 0) {
            fs::remove(temporary);
            if (!fetch_range(request, 0, range.size, temporary, reported_size, error, true)) goto fail;
        } else if (!fetch_range(request, range.offset, range.size, temporary,
                reported_size, error, true)) goto fail;
        if (reported_size != logical_size || !dan::platform::sha256_file(temporary,
                range.sha256, error) || !copy_into(temporary, request.path, range.offset, error)) goto fail;
        stats.downloaded_bytes += range.size;
        fs::remove(temporary);
    }
    stats.physical_bytes = physical_size(request.path);
    if (stats.physical_bytes == 0 || stats.physical_bytes >= stats.logical_bytes) {
        error = "model file is not sparse"; goto fail;
    }
    if (!write_sidecar(request, stats, ranges, error)) goto fail;
    return true;

fail:
    fs::remove(temporary);
    fs::remove(request.path);
    fs::remove(sidecar_path(request.path));
    return false;
}

} // namespace dan::provider_owned
