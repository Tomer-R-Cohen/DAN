#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dan::provider_owned {

template <typename T>
class FairQueue {
public:
    explicit FairQueue(std::size_t capacity) : capacity_(capacity) {}

    bool push(std::uint64_t key, T value) {
        std::lock_guard lock(mutex_);
        if (closed_ || size_ >= capacity_) return false;
        auto& queue = queues_[key];
        if (queue.empty()) ready_.push_back(key);
        queue.push_back(std::move(value));
        ++size_;
        changed_.notify_one();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [this] { return closed_ || size_ != 0; });
        while (!ready_.empty()) {
            const std::uint64_t key = ready_.front();
            ready_.pop_front();
            auto found = queues_.find(key);
            if (found == queues_.end() || found->second.empty()) continue;
            T value = std::move(found->second.front());
            found->second.pop_front();
            --size_;
            if (found->second.empty()) queues_.erase(found);
            else ready_.push_back(key);
            return value;
        }
        return std::nullopt;
    }

    template <typename Predicate>
    std::optional<T> remove_if(Predicate predicate) {
        std::lock_guard lock(mutex_);
        // ponytail: bounded linear cancellation scan; add indexing only if queue limits grow.
        for (auto map_it = queues_.begin(); map_it != queues_.end(); ++map_it) {
            auto& queue = map_it->second;
            for (auto item = queue.begin(); item != queue.end(); ++item) {
                if (!predicate(*item)) continue;
                T value = std::move(*item);
                queue.erase(item);
                --size_;
                if (queue.empty()) {
                    const std::uint64_t key = map_it->first;
                    queues_.erase(map_it);
                    std::erase(ready_, key);
                }
                return value;
            }
        }
        return std::nullopt;
    }

    std::vector<T> close() {
        std::lock_guard lock(mutex_);
        closed_ = true;
        std::vector<T> pending;
        pending.reserve(size_);
        for (auto& [key, queue] : queues_) {
            (void) key;
            while (!queue.empty()) {
                pending.push_back(std::move(queue.front()));
                queue.pop_front();
            }
        }
        queues_.clear();
        ready_.clear();
        size_ = 0;
        changed_.notify_all();
        return pending;
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return size_;
    }

    std::unordered_map<std::uint64_t, std::size_t> depths() const {
        std::lock_guard lock(mutex_);
        std::unordered_map<std::uint64_t, std::size_t> result;
        for (const auto& [key, queue] : queues_) result.emplace(key, queue.size());
        return result;
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::unordered_map<std::uint64_t, std::deque<T>> queues_;
    std::deque<std::uint64_t> ready_;
    std::size_t size_ = 0;
    bool closed_ = false;
};

} // namespace dan::provider_owned
