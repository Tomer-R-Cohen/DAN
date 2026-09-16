#pragma once
#include <string>
#include <string_view>
#include <vector>
namespace dan {
struct DistributedConfig { std::string id, executable, model, endpoints, tensor_split; };
std::vector<std::string> split_rpc_endpoints(std::string_view endpoints);
bool run_distributed_inference(const DistributedConfig&, const std::string& prompt, std::string& response);
}
