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
    }
    return 0;
}
