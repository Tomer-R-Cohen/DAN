#include "distributed_runtime.hpp"

#include <string_view>
#include <vector>

namespace dan {

std::vector<std::string> split_rpc_endpoints(std::string_view value)
{
    std::vector<std::string> result;
    while (!value.empty()) {
        const auto comma = value.find(',');
        const auto item = value.substr(0, comma);
        if (!item.empty()) result.emplace_back(item);
        if (comma == std::string_view::npos) break;
        value.remove_prefix(comma + 1);
    }
    return result;
}

bool run_distributed_inference(const DistributedConfig&, const std::string&, std::string&)
{
    return false;
}

} // namespace dan
