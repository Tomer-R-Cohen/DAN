#include "provider_ui.hpp"

#include <cassert>

int main()
{
    using dan::ProviderUiStatus;
    assert(dan::provider_status_label(ProviderUiStatus::available) == "Available");
    assert(dan::provider_status_label(ProviderUiStatus::reconnecting) == "Reconnecting");
    assert(dan::provider_status_label(ProviderUiStatus::waiting_gpu) == "Waiting for GPU resources");
    assert(dan::provider_status_label(ProviderUiStatus::recovering) == "Recovering");
    assert(dan::provider_status_label(ProviderUiStatus::action_required) == "Action required");
    assert(dan::format_token_count(221) == "221");
    assert(dan::format_token_count(1204882) == "1,204,882");
    dan::ProviderUiState state{"NVIDIA RTX",8192,6656,ProviderUiStatus::available,true};
    auto compact = dan::render_provider_dashboard(state, 45, false);
    assert(compact.find("Available") != std::string::npos && compact.find("\x1b") == std::string::npos);
    state.status=ProviderUiStatus::downloading; state.download_percent=73; state.model_name="Model";
    auto preparing=dan::render_provider_dashboard(state,70,false,2);
    assert(preparing.find("Downloading") != std::string::npos);
    assert(preparing.find("73%") != std::string::npos);
    state.status=ProviderUiStatus::contributing; state.tokens_participated=12481;
    auto ready=dan::render_provider_dashboard(state,70,false);
    assert(ready.find("Contributing") != std::string::npos);
    assert(ready.find("12,481 tokens participated in") != std::string::npos);
    assert(ready.find("12,481 tokens participated in",
        ready.find("12,481 tokens participated in") + 1) == std::string::npos);
    state.status=ProviderUiStatus::action_required; state.diagnostics="provider.log";
    auto failed=dan::render_provider_dashboard(state,70,false);
    assert(failed.find("Action required") != std::string::npos);
    assert(failed.find("Diagnostics: provider.log") != std::string::npos);
    assert(dan::render_provider_dashboard(state,45,false).find("Diagnostics: provider.log")
        != std::string::npos);
}
