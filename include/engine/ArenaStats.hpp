#pragma once

#include "core/Action.hpp"
#include "core/Types.hpp"
#include "joker_c_api.h"
#include <array>
#include <cstdint>
#include <span>

namespace joker::engine {

    struct alignas(64) SlotStats {
        std::array<JokerPlayerStats, core::kNumPlayers> players{};
        std::array<uint8_t, core::kNumPlayers> matchBonusesCount{};

        void Reset() noexcept;
    };

    void RecordSlotStepStats(
        SlotStats& ss,
        core::RulePreset preset,
        core::MatchMode mode,
        core::TeamMode teamMode,
        core::Phase prevPhase,
        core::PlayerId actor,
        core::ActionId action,
        int prevDeal,
        int prevCardsInDeal,
        const std::array<int8_t, core::kNumPlayers>& prevBids,
        const std::array<int8_t, core::kNumPlayers>& finalDealTricks,
        const std::array<int32_t, core::kNumPlayers>& prevScores,
        const std::array<int32_t, core::kNumPlayers>& afterScores,
        const std::array<bool, core::kNumPlayers>& prevPulkaClean,
        bool dealDone,
        bool matchDone) noexcept;

    void AggregateSlotStats(
        std::span<const SlotStats> slotStats,
        std::span<JokerPlayerStats, core::kNumPlayers> outStats) noexcept;

} // namespace joker::engine