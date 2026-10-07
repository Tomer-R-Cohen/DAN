// Speed measurements (docs/BETA_SELECTION_PLAN.md, M3): buckets, the bounded table, the wire
// form, hostile values and how planners pick a comparable measurement. Executable in Release.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "provider_owned/formation.hpp"
#include "provider_owned/measurement.hpp"

#include <cassert>
#include <string>

namespace po = dan::provider_owned;

int main() {
    const std::string sha(64, 'a');
    const std::string other(64, 'b');
    constexpr std::int64_t now = 1'800'000'000;

    // Buckets: powers of two, so 16K and 32K are never mixed, nor 1 and 4 sessions.
    assert(po::context_bucket(1) == 512 && po::context_bucket(512) == 512);
    assert(po::context_bucket(16384) == 16384 && po::context_bucket(16385) == 32768);
    assert(po::sessions_bucket(1) == 1 && po::sessions_bucket(3) == 4 && po::sessions_bucket(500) == 64);

    // Link bandwidth from large frames: small frames and nonsense are ignored, recent frames
    // count most, entries age out and the table stays bounded.
    {
        po::BandwidthTable table;
        table.observe("peer", 512 * 1024, 0.1, now);          // below 1 MiB: noise
        table.observe("peer", 4 << 20, 0, now);               // no time
        table.observe("", 4 << 20, 1, now);                   // no peer
        assert(table.kib_per_s("peer", now) == 0);
        table.observe("peer", 4 << 20, 1.0, now);             // 4 MiB in 1 s
        assert(table.kib_per_s("peer", now) == 4096);
        table.observe("peer", 4 << 20, 0.5, now + 1);         // 8 MiB/s: 0.7 * 4096 + 0.3 * 8192
        assert(table.kib_per_s("peer", now + 1) == 5324);
        assert(table.kib_per_s("peer", now + 1 + po::bandwidth_max_age_s + 1) == 0);
        table.observe("peer", 2 << 20, 1.0, now + 2 * po::bandwidth_max_age_s);  // stale: replaced
        assert(table.kib_per_s("peer", now + 2 * po::bandwidth_max_age_s) == 2048);
        for (std::size_t index = 0; index < po::max_bandwidth_peers; ++index) {
            table.observe("p" + std::to_string(index), 1 << 20, 1.0, now + 3 * po::bandwidth_max_age_s);
        }
        assert(table.kib_per_s("peer", now + 3 * po::bandwidth_max_age_s) == 0);  // oldest dropped
        assert(table.kib_per_s("p0", now + 3 * po::bandwidth_max_age_s) == 1024);
    }

    // The table averages, gives old evidence a bounded weight, and stays bounded.
    {
        po::SpeedTable table;
        const po::SpeedKey key{sha, po::Phase::decode, 32768, 1};
        table.observe(key, 1000, 100, now);
        table.observe(key, 2000, 32, now + 10);
        assert(table.records().size() == 1);
        assert(table.records()[0].us_per_gib == 1500);  // 32 old (capped) and 32 new
        assert(table.records()[0].samples == 132 && table.records()[0].updated_unix_s == now + 10);
        table.observe(key, 0, 5, now);                  // nonsense is ignored
        table.observe(key, 1e12, 5, now);
        table.observe(key, 1000, 0, now);
        assert(table.records()[0].samples == 132);
        // 84 distinct configurations: the oldest are dropped, the first one among them.
        for (std::uint32_t index = 0; index < 84; ++index) {
            table.observe({other, po::Phase::decode, 512u << (index % 12), 1u << (index / 12)},
                10, 1, now + 100 + index);
        }
        assert(table.records().size() == po::max_speed_records);
        assert(std::none_of(table.records().begin(), table.records().end(),
            [&](const po::SpeedRecord& record) { return record.key == key; }));
    }

    // Wire form round-trips; ages, not clocks, cross the network.
    {
        const po::SpeedRecord record{{sha, po::Phase::prefill, 16384, 2}, 812.4, 40, now - 60};
        const std::string text = po::speed_text(record, now);
        assert(text == sha + ":p:16384:2:812:40:60");
        po::SpeedRecord parsed;
        assert(po::parse_speed_text(text, parsed, now + 1000));
        assert(parsed.key == record.key && parsed.us_per_gib == 812 && parsed.samples == 40);
        assert(parsed.updated_unix_s == now + 1000 - 60);
        // Hostile or malformed entries are refused.
        for (const std::string bad : {sha + ":p:16384:2:812:40", sha + ":p:16384:2:812:40:60:1",
                sha + ":x:16384:2:812:40:60", sha + ":p:16000:2:812:40:60", sha + ":p:16384:3:812:40:60",
                sha + ":p:16384:2:0:40:60", sha + ":p:16384:2:9999999999:40:60",
                sha + ":p:16384:2:812:0:60", sha + ":p:16384:2:-5:40:60", sha + ":p:16384:2:812:40:x",
                std::string(64, 'A') + ":p:16384:2:812:40:60", std::string(63, 'a') + ":p:512:1:1:1:1",
                sha + ":p:16384:2:812:40:99999999999"}) {
            assert(!po::parse_speed_text(bad, parsed, now));
        }
    }

    // Estimates: exact first; heavier conditions are an acceptable upper bound; lighter ones,
    // other models, stale entries and thin evidence are not proof.
    {
        const po::SpeedKey wanted{sha, po::Phase::decode, 16384, 2};
        std::vector<po::SpeedRecord> records{
            {{sha, po::Phase::decode, 8192, 2}, 100, 500, now},   // lighter context: ignored
            {{sha, po::Phase::decode, 65536, 4}, 900, 500, now},  // heavier: usable
            {{sha, po::Phase::prefill, 16384, 2}, 50, 500, now},  // other phase: ignored
            {{other, po::Phase::decode, 16384, 2}, 50, 500, now}, // other model: ignored
        };
        auto estimate = po::estimate_speed(records, wanted, now, 4000);
        assert(estimate.measured && estimate.us_per_gib == 900);
        records.push_back({{sha, po::Phase::decode, 32768, 2}, 700, 500, now});
        estimate = po::estimate_speed(records, wanted, now, 4000);
        assert(estimate.measured && estimate.us_per_gib == 700);  // the closest heavier one
        records.push_back({{sha, po::Phase::decode, 16384, 2}, 600, 5, now});
        estimate = po::estimate_speed(records, wanted, now, 4000);
        assert(estimate.measured && estimate.us_per_gib == 700);  // exact but 5 samples: noise
        estimate = po::estimate_speed({records.back()}, wanted, now, 4000);
        assert(!estimate.measured && estimate.us_per_gib == 4000);
        estimate = po::estimate_speed(records, wanted, now + po::speed_max_age_s + 1, 4000);
        assert(!estimate.measured && estimate.us_per_gib == 4000);  // everything is stale
        estimate = po::estimate_speed({}, wanted, now, 0);
        assert(!estimate.measured && estimate.us_per_gib == 0);  // unknown, not zero cost
    }

    // Greetings carry measurements; malformed ones or too many reject the whole greeting.
    {
        po::ProviderCapability hello;
        hello.id = "w";
        hello.gpu = "gpu";
        hello.offered_vram_mib = 4096;
        hello.ring_endpoint = "127.0.0.1:7000";
        const std::int64_t wall = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        hello.speeds.push_back({{sha, po::Phase::decode, 32768, 1}, 5000, 64, wall - 30});
        po::ProviderCapability parsed;
        assert(po::parse_available(po::available_message(hello), parsed));
        assert(parsed.speeds.size() == 1 && parsed.speeds[0].key == hello.speeds[0].key);
        const std::string base = po::available_message(hello);
        assert(!po::parse_available(base + "\nmeasured=" + sha + ":d:32768:1:0:64:30", parsed));
        std::string flood = base;
        for (std::size_t index = 0; index < po::max_speed_records; ++index) {
            flood += "\nmeasured=" + sha + ":d:512:1:10:1:1";
        }
        assert(!po::parse_available(flood, parsed));

        // Hostile greetings (selection plan M7): implausible values or floods are refused
        // whole, never clamped.
        const std::string model_line = "\nmodel=" + sha;
        std::string many_models = base, many_ranges = base;
        for (std::size_t index = 0; index <= po::max_greeting_models; ++index) many_models += model_line;
        for (std::size_t index = 0; index <= po::max_greeting_cached; ++index) {
            many_ranges += "\ncached=" + sha + ":0-1";
        }
        for (const std::string& hostile : {
                base + "\nvram_mib=99999999999",
                base + "\nmax_context=4294967295",
                base + "\nmax_sessions=100000",
                base + "\nidle_s=5\nopen_sessions=100000",
                base + "\nspeed=99999999999",
                base + "\ncached=" + sha + ":0-5000",
                base + "\nid=" + std::string(200, 'x'),
                base + "\ngpu=" + std::string(200, 'x'),
                base + "\nabi=" + std::string(200, 'x'),
                base + "\nx=" + std::string(po::max_greeting_bytes, 'y'),
                many_models, many_ranges}) {
            assert(!po::parse_available(hostile, parsed));
        }
        // Measured links (M3): parsed and bounded.
        const std::string peer = "12D3KooWLRPJAA5o6Jip5BRHM1u5f2jpnztFoo2Hqvm6aCmbKW8C";
        assert(po::parse_available(base + "\nlink=" + peer + ":42:relay", parsed));
        assert(parsed.links.size() == 1 && parsed.links[0].rtt_ms == 42 && parsed.links[0].relayed);
        po::ProviderCapability with_link = parsed;
        assert(po::parse_available(po::available_message(with_link), parsed) && parsed.links.size() == 1);
        std::string many_links = base;
        for (std::size_t index = 0; index <= po::max_greeting_links; ++index) many_links += "\nlink=" + peer + ":1:direct";
        for (const std::string& hostile : {base + "\nlink=" + peer + ":99999:direct",
                base + "\nlink=" + peer + ":5:sideways", base + "\nlink=not-a-peer:5:direct",
                base + "\nlink=" + peer + ":5", many_links}) {
            assert(!po::parse_available(hostile, parsed));
        }
        // Link bandwidth: its own key after the link= line, so older parsers skip it.
        assert(po::parse_available(base + "\nlink=" + peer + ":42:direct\nlink_bw=" + peer + ":5120", parsed));
        assert(parsed.links.size() == 1 && parsed.links[0].kib_per_s == 5120);
        const std::string round_trip = po::available_message(parsed);
        assert(round_trip.find("\nlink_bw=" + peer + ":5120") != std::string::npos);
        assert(po::parse_available(round_trip, parsed) && parsed.links[0].kib_per_s == 5120);
        for (const std::string& hostile : {base + "\nlink_bw=" + peer + ":5120",  // no link= line
                base + "\nlink=" + peer + ":42:direct\nlink_bw=" + peer + ":0",
                base + "\nlink=" + peer + ":42:direct\nlink_bw=" + peer + ":999999999999",
                base + "\nlink=" + peer + ":42:direct\nlink_bw=" + peer + ":5\nlink_bw=" + peer + ":6",
                base + "\nlink=" + peer + ":42:direct\nlink_bw=" + peer}) {
            assert(!po::parse_available(hostile, parsed));
        }
        // Within bounds still parses (an unknown key is skipped for forward compatibility).
        assert(po::parse_available(base + "\nmax_context=262144\nmax_sessions=8\nnewer_field=1", parsed));
    }
    return 0;
}
