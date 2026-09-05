#include "distributed_runtime.hpp"
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cctype>
#include <string_view>
#include <vector>
namespace dan {
namespace {
void trim(std::string& s) { while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back(); std::size_t n=0; while(n<s.size()&&std::isspace(static_cast<unsigned char>(s[n])))++n; s.erase(0,n); }
}
std::vector<std::string> split_rpc_endpoints(std::string_view value) { std::vector<std::string> out; while(!value.empty()){auto comma=value.find(','); auto item=value.substr(0,comma); if(!item.empty())out.emplace_back(item); if(comma==std::string_view::npos)break; value.remove_prefix(comma+1);} return out; }
bool run_distributed_inference(const DistributedConfig& c, const std::string& prompt, std::string& response) {
    std::string devices; for(std::size_t i=0;i<split_rpc_endpoints(c.endpoints).size();++i){if(!devices.empty())devices+=',';devices+="RPC"+std::to_string(i);}
    int pipefd[2]; if(pipe(pipefd)==-1)return false; pid_t pid=fork(); if(pid==-1){close(pipefd[0]);close(pipefd[1]);return false;}
    if(pid==0){close(pipefd[0]);if(dup2(pipefd[1],STDOUT_FILENO)==-1)_exit(127);close(pipefd[1]);std::vector<char*> a={const_cast<char*>(c.executable.c_str()),const_cast<char*>("--model"),const_cast<char*>(c.model.c_str()),const_cast<char*>("--prompt"),const_cast<char*>(prompt.c_str()),const_cast<char*>("--rpc"),const_cast<char*>(c.endpoints.c_str()),const_cast<char*>("--device"),const_cast<char*>(devices.c_str()),const_cast<char*>("--n-gpu-layers"),const_cast<char*>("99"),const_cast<char*>("--n-predict"),const_cast<char*>("256"),const_cast<char*>("--no-conversation"),const_cast<char*>("--single-turn"),const_cast<char*>("--simple-io"),const_cast<char*>("--no-display-prompt"),const_cast<char*>("--color"),const_cast<char*>("off")};if(!c.tensor_split.empty()&&c.tensor_split!="auto"){a.push_back(const_cast<char*>("--tensor-split"));a.push_back(const_cast<char*>(c.tensor_split.c_str()));}a.push_back(nullptr);execvp(c.executable.c_str(),a.data());_exit(127);}
    close(pipefd[1]);char buf[4096];for(;;){ssize_t n=read(pipefd[0],buf,sizeof(buf));if(n>0)response.append(buf,static_cast<std::size_t>(n));else if(n==-1&&errno==EINTR)continue;else break;}close(pipefd[0]);int status=0;while(waitpid(pid,&status,0)==-1&&errno==EINTR){}trim(response);constexpr std::string_view marker="[end of text]";if(response.ends_with(marker)){response.resize(response.size()-marker.size());trim(response);}return WIFEXITED(status)&&WEXITSTATUS(status)==0&&!response.empty();
}
}
