#pragma once

#include "domain.hpp"
#include <atomic>
#include <cassert>
#include <cstddef>
#include <thread>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace conc {
template<typename T>
using default_domain = hazard_domain<T>;

template <typename T, typename domain = default_domain<T>>
requires(std::is_nothrow_destructible_v<T>)
class hazard_pointer {
    explicit hazard_pointer(domain_cell<T>* cell) noexcept : m_cell(cell) {}

   public:
    [[nodiscard]]
    static hazard_pointer make_hazard_pointer() noexcept {
        return hazard_pointer(s_domain.capture_cell());
    }

    static void retire(T* data) {
        s_domain.retire(data);
    }

    struct guard {
        guard() = delete;
        guard(const guard&) = delete;
        guard(guard&&) = delete;

        explicit guard(hazard_pointer& hp, const std::atomic<T*>& src) noexcept
            : m_hp(hp), m_ptr(hp.protect(src)) {}

        ~guard() {
            m_hp.reset_protection();
        }

        T* get() const noexcept { return m_ptr; }
        T* operator->() const noexcept { return m_ptr; }
        T& operator*() const noexcept { return *m_ptr; }
        explicit operator bool() const noexcept { return m_ptr != nullptr; }

       private:
        hazard_pointer& m_hp;
        T* m_ptr;
    };

    hazard_pointer() noexcept = default;

    hazard_pointer(hazard_pointer&& hp) noexcept : m_cell(std::exchange(hp.m_cell, nullptr)) {}

    hazard_pointer& operator=(hazard_pointer&& hp) noexcept {
        if (this != &hp) {
            if (m_cell) {
                s_domain.release_cell(m_cell);
            }
            m_cell = std::exchange(hp.m_cell, nullptr);
        }
        return *this;
    }

    ~hazard_pointer() {
        if (m_cell) {
            s_domain.release_cell(m_cell);
        }
    }

    [[nodiscard]]
    bool empty() const noexcept {
        if (!m_cell)
            return true;
        auto* ptr = m_cell->pointer.load(std::memory_order_relaxed);

        return ptr == nullptr || ptr == domain::reserved::reset();
    }

    T* protect(const std::atomic<T*>& src) noexcept {
        auto* ptr = src.load(std::memory_order_relaxed);
        while (!try_protect(ptr, src)) {
            #if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
            #elif defined(__aarch64__)
            // i dont really have iphone to test this, but it should work
            __asm__ __volatile__("yield");
            #else
            std::this_thread::yield();
            #endif
        }
        return ptr;
    }

    bool try_protect(T*& ptr, const std::atomic<T*>& src) noexcept {
        assert(m_cell != nullptr);

        m_cell->pointer.store(ptr, std::memory_order_seq_cst);

        auto* current = src.load(std::memory_order_acquire);
        if (current == ptr) {
            return true;
        }

        m_cell->pointer.store(domain::reserved::reset(), std::memory_order_release);
        ptr = current;
        return false;
    }

    void reset_protection() noexcept {
        assert(m_cell != nullptr);
        m_cell->pointer.store(domain::reserved::reset(), std::memory_order_release);
    }

    void reset_protection(T* ptr) noexcept {
        assert(m_cell != nullptr);
        m_cell->pointer.store(ptr, std::memory_order_seq_cst);
    }

    void reset_protection(std::nullptr_t) noexcept {
        reset_protection();
    }

    void swap(hazard_pointer& t) noexcept {
        std::swap(m_cell, t.m_cell);
    }

   private:
    inline static auto s_domain = domain();
    domain_cell<T>* m_cell = nullptr;
};

template<typename T, typename domain>
void swap(hazard_pointer<T, domain>& t1, hazard_pointer<T, domain>& t2) noexcept {
    t1.swap(t2);
}
}