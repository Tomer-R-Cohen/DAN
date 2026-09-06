#include "admin_dashboard.hpp"
#include "control_plane.hpp"

#include <cassert>

int main()
{
    std::vector<dan::AdminProvider> providers{
        {"assigned", "a", "GPU A", "ASSIGNED", "READY", 8192, 6656, 42, true, 1},
        {"spare", "b", "GPU B", "SPARE", "AVAILABLE", 12288, 10752, 0, true, 2},
        {"offline", "c", "GPU C", "OFFLINE", "OFFLINE", 24576, 23040, 9, false, 0}};
    const auto totals = dan::calculate_admin_totals(providers);
    assert(totals.online == 2 && totals.assigned == 1 && totals.spare == 1 && totals.offline == 1);
    assert(totals.physical_vram_mib == 20480 && totals.offered_vram_mib == 17408);
    assert(totals.assigned_vram_mib == 6656 && totals.spare_vram_mib == 10752);
    assert(dan::parse_generated_tokens("llamacpp:tokens_predicted_total 184392\n") == 184392);
    assert(dan::parse_completed_requests("llamacpp:requests_total 438\n") == 438);
    assert(!dan::artifact_fits_disk(5000, 4999) && dan::artifact_fits_disk(5000, 5000));
    dan::ShardMetadata shard; shard.size_bytes=4ULL*1024*1024*1024; shard.min_vram_mib=1024;
    assert(dan::required_current_vram_mib(shard) == 4608);
    dan::EventTimeline timeline(2); timeline.add("one"); timeline.add("two"); timeline.add("three");
    assert(timeline.entries().size() == 2 && timeline.entries().front().text == "three");
    dan::AdminSnapshot snapshot; snapshot.model_name="SmolLM2"; snapshot.model_id="dan-main";
    snapshot.model_version="v1"; snapshot.quantization="Q4_K_M"; snapshot.replicas_total=1;
    snapshot.replicas_ready=1; snapshot.providers=providers;
    const std::string json=dan::admin_snapshot_json(snapshot);
    assert(json.find("\"model_name\":\"SmolLM2\"") != std::string::npos);
    assert(json.find("\"replicas_ready\":1") != std::string::npos);
}
