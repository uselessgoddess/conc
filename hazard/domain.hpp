#pragma once

#include <atomic>
#include <type_traits>
#include <vector>
#include <array>
#include <cassert>
#include <memory>
#include <cstdint>
#include <algorithm>

#include <allocator.hpp>

namespace conc {

template<typename T>
struct alignas(CACHELINE_SIZE) domain_cell {
    std::atomic<T*> pointer = nullptr;
};

struct default_placeholder {};

template<typename T, std::size_t max_objects = 128, typename _ = default_placeholder>
requires(std::is_nothrow_destructible_v<T>)
struct hazard_domain {
    struct reserved { // im very lazy.nvim to declare friend class
        static T* sentinel() noexcept {
            return reinterpret_cast<T*>(~static_cast<std::uintptr_t>(0));
        }
        static T* reset() noexcept { // pretty new state
            return reinterpret_cast<T*>(~static_cast<std::uintptr_t>(0) - 1);
        }
    };

    hazard_domain() noexcept {
        if (!m_instance)
            m_instance = this;
    }

    [[nodiscard]]
    domain_cell<T>* capture_cell() noexcept {
        for (auto& cell : m_acquire_list) {
            T* expected = nullptr;
            if (cell.pointer.load(std::memory_order_relaxed) == nullptr &&
                cell.pointer.compare_exchange_strong(expected, reserved::sentinel(), std::memory_order_acq_rel)) {
                return &cell;
            }
        }
        // todo! avoid `abort` or break everything with unreachable
        std::abort();
    }

    void release_cell(domain_cell<T>* cell) noexcept {
        if (cell) {
            cell->pointer.store(nullptr, std::memory_order_release);
        }
    }

    void retire(T* data) {
        auto& retires = tl_retires();

        retires.push_back(data);
        if (retires.size() >= 2 * max_objects) {
            delete_hazards(retires);
        }
    }

   private:
    void delete_hazards(std::vector<T*>& buffer) noexcept {
        auto it = buffer.begin();
        auto end = buffer.end();

        while (it != end) {
            if (scan_for_hazard(*it)) {
                ++it;
            } else {
                delete *it;
                --end;
                if (it != end) {
                    *it = std::move(*end);
                }
            }
        }

        buffer.erase(end, buffer.end());
    }

    [[nodiscard]]
    bool scan_for_hazard(T* ptr) noexcept {
        auto* sentinel = reserved::sentinel();
        auto* reset = reserved::reset();

        for (const auto& cell : m_acquire_list) {
            T* hazard = cell.pointer.load(std::memory_order_seq_cst);
            if (hazard != nullptr && hazard != sentinel &&
                hazard != reset && hazard == ptr) [[unlikely]] {
                return true;
            }
        }
        return false;
    }

    static auto tl_retires() -> std::vector<T*>& {
        thread_local auto retires = retire_buffer(m_instance);
        return retires.buffer;
    }

    struct retire_buffer {
        std::vector<T*> buffer;
        hazard_domain* domain_ref = nullptr;

        explicit retire_buffer(hazard_domain* d) : domain_ref(d) {
            buffer.reserve(max_objects * 3);
        }

        ~retire_buffer() {
            if (domain_ref && !buffer.empty()) {
                domain_ref->delete_hazards(buffer);
            }
        }
    };

    inline static std::array<domain_cell<T>, max_objects> m_acquire_list;
    inline static hazard_domain* m_instance = nullptr; // more portable but still broekn on mingw
};
}