#include "admin_dashboard.hpp"
#include "control_plane.hpp"

#include <cassert>

int main()
{
    std::vector<dan::AdminProvider> providers{
        {"assigned", "a", "GPU A", "ASSIGNED", "DOWNLOADING", 6656, 768, 42, true, 1,
            2400000000, 3300000000, 18600000},
        {"spare", "b", "GPU B", "SPARE", "AVAILABLE", 10752, 0, 0, true, 2},
        {"offline", "c", "GPU C", "OFFLINE", "OFFLINE", 23040, 100, 9, false, 0}};
    const auto totals = dan::calculate_admin_totals(providers);
    assert(totals.online == 2 && totals.assigned == 1 && totals.spare == 1 && totals.offline == 1);
    assert(totals.offered_vram_mib == 17408 && totals.used_vram_mib == 768);
    std::size_t used_vram_mib = 0;
    assert(dan::parse_heartbeat_message("HEARTBEAT\nused_vram_mib=768", used_vram_mib));
    assert(used_vram_mib == 768);
    assert(!dan::parse_heartbeat_message("HEARTBEAT\nused_vram_mib=nope", used_vram_mib));
    std::size_t downloaded = 0, total = 0, speed = 0;
    const std::string progress = dan::download_progress_message(2400000000, 3300000000, 18600000);
    assert(dan::parse_download_progress_message(progress, downloaded, total, speed));
    assert(downloaded == 2400000000 && total == 3300000000 && speed == 18600000);
    assert(!dan::parse_download_progress_message(
        "DOWNLOAD_PROGRESS\ndownloaded_bytes=4\ntotal_bytes=3\nbytes_per_second=1",
        downloaded, total, speed));
    assert(dan::parse_generated_tokens("llamacpp:tokens_predicted_total 184392\n") == 184392);
    assert(dan::parse_completed_requests("llamacpp:requests_total 438\n") == 438);
    assert(!dan::artifact_fits_disk(5000, 4999) && dan::artifact_fits_disk(5000, 5000));
    dan::ShardMetadata shard; shard.size_bytes=4ULL*1024*1024*1024; shard.min_vram_mib=1024;
    assert(dan::required_current_vram_mib(shard) == 1024);
    dan::EventTimeline timeline(2); timeline.add("one"); timeline.add("two"); timeline.add("three");
    assert(timeline.entries().size() == 2 && timeline.entries().front().text == "three");
    dan::AdminSnapshot snapshot; snapshot.model_name="SmolLM2"; snapshot.model_id="dan-main";
    snapshot.model_version="v1"; snapshot.quantization="Q4_K_M"; snapshot.replicas_total=1;
    snapshot.replicas_ready=1; snapshot.providers=providers;
    const std::string json=dan::admin_snapshot_json(snapshot);
    assert(json.find("\"model_name\":\"SmolLM2\"") != std::string::npos);
    assert(json.find("\"replicas_ready\":1") != std::string::npos);
    assert(json.find("\"used_vram_mib\":768") != std::string::npos);
    assert(json.find("\"downloaded_bytes\":2400000000") != std::string::npos);
    assert(json.find("\"download_bytes_per_second\":18600000") != std::string::npos);
}
