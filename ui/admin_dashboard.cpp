#include "admin_dashboard.hpp"
#include "protocol.hpp"

#ifdef _WIN32
#include <ws2tcpip.h>
#else
#include <netdb.h>
#endif

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace dan {
namespace {
std::string escape(std::string_view value)
{
    std::string result;
    for (const char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += c; }
        else if (c == '\n') result += "\\n";
        else if (static_cast<unsigned char>(c) >= 0x20) result += c;
    }
    return result;
}

const char page[] = R"HTML(<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DAN Admin</title><style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#172033;background:#f3f5f8}*{box-sizing:border-box}body{margin:0}.wrap{max-width:1180px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}h1{margin:0;font-size:30px}.sub{color:#687386;margin-top:4px}.badge,.dot{display:inline-block;border-radius:999px}.badge{padding:8px 13px;font-size:13px;font-weight:700}.green{background:#e4f6eb;color:#177342}.yellow{background:#fff3d6;color:#8b6100}.red{background:#fde8e8;color:#a92b2b}.gray{background:#e9edf2;color:#596579}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:#fff;border:1px solid #e3e7ed;border-radius:14px;padding:18px;box-shadow:0 4px 18px #25324a0a}.metric{font-size:27px;font-weight:750;margin-top:8px}.label{font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:#778296}.wide{grid-column:span 2}.model{display:flex;gap:18px;align-items:center}.art{width:94px;height:94px;border-radius:18px;background:linear-gradient(145deg,#25324a,#55708f);display:grid;place-items:center}.art svg{width:57px}.section{margin-top:18px}h2{font-size:18px;margin:0 0 13px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid #edf0f4;font-size:14px}th{color:#758094;font-size:11px;text-transform:uppercase}.muted{opacity:.55}.download{width:180px;height:7px;background:#e8ecf1;border-radius:5px;margin:8px 0 5px;overflow:hidden}.download i{display:block;height:100%;background:#3d78c5}.timeline{max-height:280px;overflow:auto}.event{display:grid;grid-template-columns:70px 1fr;gap:12px;padding:8px 0;border-bottom:1px solid #edf0f4;font-size:14px}.event time{color:#7c8798}.actions{display:flex;gap:10px}.button{color:white;background:#253b58;text-decoration:none;padding:10px 15px;border-radius:9px;font-weight:650}.pulse{width:9px;height:9px;background:#29a35a;margin-right:7px;animation:p 1.8s infinite}@keyframes p{50%{box-shadow:0 0 0 7px #29a35a20}}@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}.wrap{padding:17px}}@media(max-width:520px){.grid{display:block}.card{margin-bottom:12px}.top{align-items:flex-start;gap:12px;flex-direction:column}}
</style></head><body><main class="wrap"><header class="top"><div><h1>DAN</h1><div class="sub">Decentralized AI Network · Testnet administration</div></div><div class="actions"><a class="button" href="http://127.0.0.1:8080" target="_blank">Open Chat</a><span id="network" class="badge gray">Starting</span></div></header><section class="grid"><div class="card"><div class="label">Providers</div><div id="providers" class="metric">—</div><div id="providerDetail" class="sub"></div></div><div class="card"><div class="label">Replicas</div><div id="replicas" class="metric">—</div><div id="replicaState" class="sub"></div></div><div class="card"><div class="label">DAN VRAM</div><div id="vram" class="metric">—</div><div id="vramDetail" class="sub"></div></div><div class="card"><div class="label">Generated</div><div id="tokens" class="metric">0</div><div id="requests" class="sub">0 requests</div></div><div class="card wide"><div class="model"><div class="art"><svg viewBox="0 0 64 64" fill="none"><circle cx="32" cy="32" r="25" stroke="#fff" stroke-width="3"/><circle cx="22" cy="28" r="4" fill="#fff"/><circle cx="42" cy="28" r="4" fill="#fff"/><path d="M20 42c8 5 16 5 24 0" stroke="#fff" stroke-width="3" stroke-linecap="round"/></svg></div><div><div class="label">Managed model</div><h2 id="modelName" style="margin:6px 0">—</h2><div id="modelMeta" class="sub"></div><div id="modelState" style="margin-top:9px"></div></div></div></div><div class="card wide"><div class="label">System readiness</div><div id="states" style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px"></div></div></section><section class="card section"><h2>Providers</h2><div style="overflow:auto"><table><thead><tr><th>PC</th><th>GPU</th><th>DAN VRAM</th><th>Role</th><th>State</th><th>Tokens participated in</th><th>Last seen</th></tr></thead><tbody id="providerRows"></tbody></table></div></section><section class="card section"><h2>Recent events</h2><div id="timeline" class="timeline"></div></section></main><script>
const fmt=n=>Number(n||0).toLocaleString(),gb=n=>(n/1024).toFixed(1)+' GB',filegb=n=>(n/1e9).toFixed(1)+' GB',speed=n=>(n/1e6).toFixed(1)+' MB/s',h=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),badge=(s)=>{let c=/READY|CONNECTED|CONTRIBUTING/.test(s)?'green':/ERROR|ACTION/.test(s)?'red':/OFFLINE/.test(s)?'gray':'yellow';return `<span class="badge ${c}">${h(s)}</span>`},download=p=>p.download_total_bytes?`<small>Downloading required files</small><div class="download"><i style="width:${Math.min(100,p.downloaded_bytes*100/p.download_total_bytes)}%"></i></div><small>${filegb(p.downloaded_bytes)} / ${filegb(p.download_total_bytes)} at ${speed(p.download_bytes_per_second)}</small>`:'';async function refresh(){try{let d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());let t=d.totals;network.innerHTML='<span class="dot pulse"></span>'+h(d.network_status);network.className='badge green';providers.textContent=t.online+' online';providerDetail.textContent=t.assigned+' assigned · '+t.spare+' spare · '+t.offline+' offline';replicas.textContent=d.replicas_ready+' / '+d.replicas_total+' ready';replicaState.textContent='Replica '+d.replica_status;vram.textContent=gb(t.used_vram_mib)+' used now';vramDetail.textContent=gb(t.offered_vram_mib)+' offered to DAN'+(d.runtime_status==='READY'?'':' · model not loaded yet');tokens.textContent=fmt(d.generated_tokens);requests.textContent=fmt(d.completed_requests)+' completed requests';modelName.textContent=d.model_name;modelMeta.textContent=`${d.model_id} · ${d.model_version} · ${d.quantization} · ${(d.model_bytes/1048576).toFixed(0)} MiB`;modelState.innerHTML=badge('Replica '+d.replica_status)+' '+badge('Runtime '+d.runtime_status);states.innerHTML=badge('Providers '+d.provider_status)+badge('Replica '+d.replica_status)+badge('Runtime '+d.runtime_status);providerRows.innerHTML=d.providers.map(p=>`<tr class="${p.online?'':'muted'}"><td><strong>${h(p.name)}</strong><br><small>${h(p.id)}</small></td><td>${h(p.gpu)}</td><td>${gb(p.used_vram_mib)} used now / ${gb(p.offered_vram_mib)} offered</td><td>${badge(p.role)}</td><td>${badge(p.state)}${download(p)}</td><td>${fmt(p.tokens_participated)}</td><td>${p.online?(p.last_seen_seconds+'s ago'):'Offline'}</td></tr>`).join('');timeline.innerHTML=d.events.map(e=>`<div class="event"><time>${h(e.time)}</time><div>${h(e.text)}</div></div>`).join('')||'<div class="sub">No events yet.</div>'}catch(e){network.textContent='Coordinator unavailable';network.className='badge red'}}refresh();setInterval(refresh,1000);
</script></body></html>)HTML";

std::string response(std::string_view type, std::string_view body, std::string_view status = "200 OK")
{
    return "HTTP/1.1 " + std::string(status) + "\r\nContent-Type: " + std::string(type)
        + "\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: "
        + std::to_string(body.size()) + "\r\n\r\n" + std::string(body);
}
}

AdminTotals calculate_admin_totals(const std::vector<AdminProvider>& providers)
{
    AdminTotals totals;
    for (const auto& p : providers) {
        if (!p.online) { ++totals.offline; continue; }
        ++totals.online; totals.offered_vram_mib += p.offered_vram_mib;
        totals.used_vram_mib += p.used_vram_mib;
        if (p.role == "ASSIGNED") ++totals.assigned;
        else ++totals.spare;
    }
    return totals;
}

std::size_t parse_metric(std::string_view metrics, const std::vector<std::string_view>& names)
{
    for (const auto name : names) {
        const auto position = metrics.find(name);
        if (position == std::string_view::npos) continue;
        auto value = metrics.substr(position + name.size());
        const auto first = value.find_first_of("0123456789");
        if (first == std::string_view::npos) continue;
        value.remove_prefix(first);
        std::size_t result = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
        if (parsed.ec == std::errc{}) return result;
    }
    return 0;
}

std::size_t parse_generated_tokens(std::string_view metrics)
{
    return parse_metric(metrics, {"llamacpp:tokens_predicted_total", "llamacpp_tokens_predicted_total"});
}

std::size_t parse_completed_requests(std::string_view metrics)
{
    return parse_metric(metrics, {"llamacpp:requests_total", "llamacpp_requests_total",
        "llamacpp:requests_completed_total", "llamacpp_requests_completed_total"});
}

void diagnostic_log(std::string_view component, std::string_view text)
{
    const auto directory = platform::data_directory() / "logs";
    std::error_code error; std::filesystem::create_directories(directory, error);
    std::ofstream log(directory / (std::string(component) + ".log"), std::ios::app);
    if (log) log << std::time(nullptr) << ' ' << text << '\n';
}

void EventTimeline::add(std::string text, std::string level)
{
    std::time_t now = std::time(nullptr); std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream time; time << std::put_time(&local, "%H:%M:%S");
    events_.push_front({time.str(), std::move(text), std::move(level)});
    while (events_.size() > limit_) events_.pop_back();
}

std::vector<AdminEvent> EventTimeline::entries() const { return {events_.begin(), events_.end()}; }

std::string admin_snapshot_json(const AdminSnapshot& s)
{
    const auto t = calculate_admin_totals(s.providers);
    std::ostringstream out;
    out << "{\"network_status\":\"" << escape(s.network_status) << "\",\"provider_status\":\"" << escape(s.provider_status)
        << "\",\"replica_status\":\"" << escape(s.replica_status) << "\",\"runtime_status\":\"" << escape(s.runtime_status)
        << "\",\"model_name\":\"" << escape(s.model_name) << "\",\"model_id\":\"" << escape(s.model_id)
        << "\",\"model_version\":\"" << escape(s.model_version) << "\",\"quantization\":\"" << escape(s.quantization)
        << "\",\"chat_url\":\"" << escape(s.chat_url) << "\",\"model_bytes\":" << s.model_bytes
        << ",\"replicas_total\":" << s.replicas_total << ",\"replicas_ready\":" << s.replicas_ready
        << ",\"generated_tokens\":" << s.generated_tokens << ",\"completed_requests\":" << s.completed_requests
        << ",\"latest_tokens_per_second\":" << s.latest_tokens_per_second
        << ",\"totals\":{\"online\":" << t.online << ",\"assigned\":" << t.assigned << ",\"spare\":" << t.spare
        << ",\"offline\":" << t.offline << ",\"offered_vram_mib\":" << t.offered_vram_mib
        << ",\"used_vram_mib\":" << t.used_vram_mib << "},\"providers\":[";
    for (std::size_t i = 0; i < s.providers.size(); ++i) {
        const auto& p = s.providers[i]; if (i) out << ',';
        out << "{\"name\":\"" << escape(p.name) << "\",\"id\":\"" << escape(p.id) << "\",\"gpu\":\"" << escape(p.gpu)
            << "\",\"role\":\"" << escape(p.role) << "\",\"state\":\"" << escape(p.state) << "\",\"offered_vram_mib\":"
            << p.offered_vram_mib << ",\"used_vram_mib\":" << p.used_vram_mib << ",\"tokens_participated\":"
            << p.tokens_participated << ",\"online\":" << (p.online ? "true" : "false") << ",\"last_seen_seconds\":" << p.last_seen_seconds
            << ",\"downloaded_bytes\":" << p.downloaded_bytes << ",\"download_total_bytes\":" << p.download_total_bytes
            << ",\"download_bytes_per_second\":" << p.download_bytes_per_second << '}';
    }
    out << "],\"events\":[";
    for (std::size_t i = 0; i < s.events.size(); ++i) { if (i) out << ','; const auto& e=s.events[i]; out << "{\"time\":\"" << escape(e.time) << "\",\"text\":\"" << escape(e.text) << "\",\"level\":\"" << escape(e.level) << "\"}"; }
    return out.str() + "]}";
}

void render_admin_terminal(const AdminSnapshot& s)
{
    platform::clear_console();
    const auto totals = calculate_admin_totals(s.providers);
    const auto color = [](std::string_view state) {
        return state == "READY" || state == "CONNECTED" ? "\x1b[32m"
            : (state == "ERROR" ? "\x1b[31m" : "\x1b[33m");
    };
    std::printf("\x1b[1mDAN\x1b[0m  Decentralized AI Network\n"
        "\x1b[90mTestnet coordinator | %s\x1b[0m\n\n", s.setup_address.c_str());
    std::printf("Network   %s%s\x1b[0m    Providers %s%s\x1b[0m    Replica %s%s\x1b[0m    Runtime %s%s\x1b[0m\n\n",
        color(s.network_status), s.network_status.c_str(), color(s.provider_status), s.provider_status.c_str(),
        color(s.replica_status), s.replica_status.c_str(), color(s.runtime_status), s.runtime_status.c_str());
    std::printf("\x1b[1mMODEL\x1b[0m\n%s  |  %s  |  %.0f MiB\n%s / %s\n\n",
        s.model_name.c_str(), s.quantization.c_str(), s.model_bytes / 1048576.0,
        s.model_id.c_str(), s.model_version.c_str());
    std::printf("\x1b[1mNETWORK\x1b[0m\nProviders: %zu online, %zu assigned, %zu spare, %zu offline\n"
        "DAN VRAM: %.1f GB used now / %.1f GB offered%s\n"
        "Generated: %zu tokens | Requests: %zu\n\n",
        totals.online, totals.assigned, totals.spare, totals.offline,
        totals.used_vram_mib/1024.0, totals.offered_vram_mib/1024.0,
        s.runtime_status == "READY" ? "" : " (model not loaded yet)",
        s.generated_tokens, s.completed_requests);
    std::printf("\x1b[1mPROVIDERS\x1b[0m\n%-18s %-28s %9s %-10s %-12s %10s\n",
        "PC", "GPU", "DAN VRAM", "ROLE", "STATE", "TOKENS");
    if (s.providers.empty()) std::printf("\x1b[90mWaiting for a provider...\x1b[0m\n");
    for (const auto& p : s.providers) {
        std::printf("%-18.18s %-28.28s %3.1f/%3.1fGB %-10.10s %-12.12s %10zu\n",
            p.name.c_str(), p.gpu.c_str(), p.used_vram_mib/1024.0, p.offered_vram_mib/1024.0,
            p.role.c_str(), p.state.c_str(), p.tokens_participated);
        if (p.download_total_bytes != 0) std::printf(
            "  Downloading required files: %.1f%% (%.1f / %.1f GB at %.1f MB/s)\n",
            p.downloaded_bytes * 100.0 / p.download_total_bytes,
            p.downloaded_bytes / 1000000000.0, p.download_total_bytes / 1000000000.0,
            p.download_bytes_per_second / 1000000.0);
    }
    std::printf("\n\x1b[1mRECENT EVENTS\x1b[0m\n");
    const std::size_t shown = std::min<std::size_t>(8, s.events.size());
    for (std::size_t i=0; i<shown; ++i) std::printf("\x1b[90m%s\x1b[0m  %s\n",
        s.events[i].time.c_str(), s.events[i].text.c_str());
    std::printf("\n\x1b[90mChat opens automatically when ready | Optional web view: http://127.0.0.1:9090\n"
        "Press Ctrl+C to stop DAN\x1b[0m\n");
}

AdminDashboard::~AdminDashboard()
{
    thread_.request_stop(); platform::shutdown_socket(listener_); platform::close_socket(listener_);
    listener_ = platform::invalid_socket;
    if (thread_.joinable()) thread_.join();
}

bool AdminDashboard::start(std::string_view port, std::string& error)
{
    addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (getaddrinfo("127.0.0.1", std::string(port).c_str(), &hints, &addresses) != 0) { error="admin address lookup failed"; return false; }
    listener_ = socket(addresses->ai_family, addresses->ai_socktype, addresses->ai_protocol);
    const int reuse = 1;
    const bool ok = listener_ != platform::invalid_socket
        && setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse)) == 0
        && bind(listener_, addresses->ai_addr, static_cast<int>(addresses->ai_addrlen)) == 0 && listen(listener_, 8) == 0;
    freeaddrinfo(addresses);
    if (!ok) { platform::close_socket(listener_); listener_=platform::invalid_socket; error="localhost admin port is unavailable"; return false; }
    thread_ = std::jthread([this](std::stop_token stop){ serve(stop); }); return true;
}

void AdminDashboard::update(std::string json) { std::lock_guard lock(mutex_); json_ = std::move(json); }

void AdminDashboard::serve(std::stop_token stop)
{
    while (!stop.stop_requested()) {
        const auto client = accept(listener_, nullptr, nullptr);
        if (client == platform::invalid_socket) break;
        char request[2048]{}; const int count = recv(client, request, sizeof(request)-1, 0);
        std::string body;
        if (count > 0 && std::string_view(request, count).starts_with("GET /api/status ")) {
            std::lock_guard lock(mutex_); body = response("application/json; charset=utf-8", json_);
        } else if (count > 0 && std::string_view(request, count).starts_with("GET / ")) {
            body = response("text/html; charset=utf-8", page);
        } else body = response("text/plain", "Not found", "404 Not Found");
        send_all(client, body.data(), body.size()); platform::close_socket(client);
    }
}

std::string http_get(std::string_view host, std::string_view port, std::string_view path)
{
    const auto socket = platform::connect_tcp(host, port); if (socket == platform::invalid_socket) return {};
    const std::string request = "GET " + std::string(path) + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    if (!send_all(socket, request.data(), request.size())) { platform::close_socket(socket); return {}; }
    std::string result; char buffer[4096]; bool failed=false;
    while (platform::wait_readable(socket, 500, failed)) { const int n=recv(socket,buffer,sizeof(buffer),0); if(n<=0) break; result.append(buffer,n); }
    platform::close_socket(socket); const auto body=result.find("\r\n\r\n"); return body==std::string::npos ? std::string{} : result.substr(body+4);
}
} // namespace dan
