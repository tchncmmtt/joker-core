#pragma once

#include "core/Action.hpp"
#include "core/GameState.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace joker::engine {

    struct DealHistoryBuffer {
        static constexpr size_t kMaxSeqLen = 48;
        static constexpr size_t kTokenDim = 8;

        struct RawEvent {
            uint8_t eventType{0};
            uint8_t absPlayer{0};
            uint8_t actionId{0};
            uint8_t cardId{255};
            uint8_t suit{4};
            uint8_t jokerAction{0};
            uint8_t trickIdx{255};
            uint8_t posInTrick{255};
        };

        std::array<RawEvent, kMaxSeqLen> events{};
        uint8_t count{0};

        constexpr void Clear() noexcept {
            count = 0;
        }

        constexpr void Push(const RawEvent& ev) noexcept {
            if (count < kMaxSeqLen) {
                events[count++] = ev;
            }
        }

        [[nodiscard]] static RawEvent MakeEvent(
            const core::GameState& stBefore,
            core::ActionId action) noexcept;

        void WriteTokens(
            core::PlayerId observer,
            int16_t* outTokens,
            uint8_t* outLength) const noexcept;
    };

} // namespace joker::engine