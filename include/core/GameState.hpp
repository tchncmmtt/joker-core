#pragma once

#include "core/Action.hpp"
#include "core/Card.hpp"
#include <array>
#include <cstdint>
#include <span>

namespace joker::core {

struct DealScheduleEntry {
    int pulkaIndex;
    int roundInPulka;
    int cardsPerPlayer;
    bool isNineCardDeal;
    bool isPulkaEnd;
};

inline constexpr std::array<DealScheduleEntry, kLongDeals> kLongSchedule = {{
    // 1-я пулька (1..8 карт)
    {0, 0, 1, false, false}, {0, 1, 2, false, false},
    {0, 2, 3, false, false}, {0, 3, 4, false, false},
    {0, 4, 5, false, false}, {0, 5, 6, false, false},
    {0, 6, 7, false, false}, {0, 7, 8, false, true},
    // 2-я пулька (девятки)
    {1, 0, 9, true,  false}, {1, 1, 9, true,  false},
    {1, 2, 9, true,  false}, {1, 3, 9, true,  true},
    // 3-я пулька (8..1 карт)
    {2, 0, 8, false, false}, {2, 1, 7, false, false},
    {2, 2, 6, false, false}, {2, 3, 5, false, false},
    {2, 4, 4, false, false}, {2, 5, 3, false, false},
    {2, 6, 2, false, false}, {2, 7, 1, false, true},
    // 4-я пулька (девятки)
    {3, 0, 9, true,  false}, {3, 1, 9, true,  false},
    {3, 2, 9, true,  false}, {3, 3, 9, true,  true}
}};

inline constexpr std::array<DealScheduleEntry, kShortDeals> kShortSchedule = {{
    // 1-я пулька (девятки)
    {0, 0, 9, true, false}, {0, 1, 9, true, false},
    {0, 2, 9, true, false}, {0, 3, 9, true, true},
    // 2-я пулька (девятки)
    {1, 0, 9, true, false}, {1, 1, 9, true, false},
    {1, 2, 9, true, false}, {1, 3, 9, true, true}
}};

constexpr int GetTotalDeals(const MatchMode mode) noexcept {
    return (mode == MatchMode::Short) ? kShortDeals : kLongDeals;
}

constexpr int GetPulkasCount(const MatchMode mode) noexcept {
    return (mode == MatchMode::Short) ? kShortPulkasCount : kLongPulkasCount;
}

constexpr const DealScheduleEntry& GetScheduleEntry(const MatchMode mode, const int dealNumber) noexcept {
    if (mode == MatchMode::Short) {
        JOKER_ASSUME(dealNumber >= 0 && dealNumber < kShortDeals);
        return kShortSchedule[dealNumber];
    }
    JOKER_ASSUME(dealNumber >= 0 && dealNumber < kLongDeals);
    return kLongSchedule[dealNumber];
}

struct GameStateView {
    RulePreset rules{RulePreset::Popular};
    MatchMode mode{MatchMode::Long};
    TeamMode teamMode{TeamMode::Solo};
    PulkaMode pulkaMode{PulkaMode::Strict};
    Phase phase{Phase::ChoosingTrump};
    PlayerId observerPlayer{0};
    PlayerId dealerPlayer{0};
    PlayerId currentPlayer{0};

    int dealNumber{0};
    int pulkaIndex{0};
    int roundInPulka{0};
    int cardsInDeal{1};

    Suit trumpSuit{Suit::None};
    CardId upcard{kInvalidCard};

    HandMask myHand{0};
    HandMask playedCardsHistory{0};

    std::array<int8_t, kNumPlayers> bids{{-1, -1, -1, -1}};
    std::array<int, kNumPlayers> tricksTaken{{0, 0, 0, 0}};
    std::array<int, kNumPlayers> totalScores{{0, 0, 0, 0}};

    std::array<bool, kNumPlayers> pulkaClean{{true, true, true, true}};
    std::array<bool, kNumPlayers> pulkaAllPasses{{true, true, true, true}};
    std::array<int, kNumPlayers> pulkaMaxScoreExclLast{{0, 0, 0, 0}};
    std::array<int, kNumPlayers> pulkaMaxNonPassScoreExclLast{{0, 0, 0, 0}};

    std::array<HandMask, kNumPlayers> impossibleCards{};
    std::array<uint8_t, kNumPlayers> voidSuits{{0, 0, 0, 0}};
    std::array<uint8_t, kNumPlayers> handCounts{{0, 0, 0, 0}};
    std::array<uint8_t, kNumPlayers> jokersPlayedBy{{0, 0, 0, 0}};

    std::array<TrickCard, kNumPlayers> currentTrick{};
    uint8_t cardsInCurrentTrick{0};

    static constexpr size_t kTensorSize = 320;

    void WriteSoloTensor(float* outBuffer) const noexcept;
    void WritePairsTensor(float* outBuffer) const noexcept;
    void WriteTensor(float* outBuffer) const noexcept;
    void WriteTensor(std::span<float, kTensorSize> outBuffer) const noexcept {
        WriteTensor(outBuffer.data());
    }
};

// Компактная структура состояния (ровно 192 байта = 3 кэш-линии CPU по 64 байта)
struct alignas(64) GameState {
    static constexpr uint8_t kObservedFlagWaitingDeal   = 1U << 0;
    static constexpr uint8_t kObservedFlagWaitingRemSix = 1U << 1;

    std::array<HandMask, kNumPlayers> hands{};                            // 32B (0..31)
    std::array<HandMask, kNumPlayers> impossibleCards{};                  // 32B (32..63)
    HandMask playedCardsHistory{0};                                       // 8B  (64..71)
    std::array<int32_t, kNumPlayers> totalScores{{0, 0, 0, 0}};           // 16B (72..87)

    std::array<int16_t, kNumPlayers> pulkaMaxScoreExclLast{{0, 0, 0, 0}};        // 8B  (88..95)
    std::array<int16_t, kNumPlayers> pulkaMaxNonPassScoreExclLast{{0, 0, 0, 0}}; // 8B  (96..103)

    std::array<TrickCard, kNumPlayers> currentTrick{};                    // 16B (104..119)

    std::array<int8_t, kNumPlayers> bids{{-1, -1, -1, -1}};               // 4B  (120..123)
    std::array<int8_t, kNumPlayers> tricksTaken{{0, 0, 0, 0}};            // 4B  (124..127)

    std::array<bool, kNumPlayers> pulkaClean{{true, true, true, true}};     // 4B (128..131)
    std::array<bool, kNumPlayers> pulkaAllPasses{{true, true, true, true}}; // 4B (132..135)

    std::array<uint8_t, kNumPlayers> handCounts{{0, 0, 0, 0}};            // 4B  (136..139)
    std::array<uint8_t, kNumPlayers> jokersPlayedBy{{0, 0, 0, 0}};        // 4B  (140..143)

    int8_t dealNumber{0};                 // 1B (144)
    int8_t cardsInDeal{1};                // 1B (145)
    int8_t bidsMadeInDeal{0};             // 1B (146)
    int8_t tricksCompletedInDeal{0};      // 1B (147)

    RulePreset rules{RulePreset::Popular}; // 1B (148)
    MatchMode mode{MatchMode::Long};       // 1B (149)
    TeamMode teamMode{TeamMode::Solo};     // 1B (150)
    PulkaMode pulkaMode{PulkaMode::Strict};// 1B (151)
    Phase phase{Phase::MatchFinished};     // 1B (152)
    PlayerId dealer{0};                    // 1B (153)
    PlayerId currentPlayer{0};             // 1B (154)
    PlayerId trickLeader{0};               // 1B (155)
    PlayerId lastTrickWinner{0};           // 1B (156)
    Suit trumpSuit{Suit::None};            // 1B (157)
    CardId upcard{kInvalidCard};           // 1B (158)
    uint8_t cardsInCurrentTrick{0};        // 1B (159)
    uint8_t knownHandsMask{0x0F};          // 1B (160)
    uint8_t observedFlags{0};              // 1B (161)
    PlayerId heroSeat{0};                  // 1B (162)

    [[nodiscard]] constexpr bool IsHandKnown(const PlayerId p) const noexcept {
        return ((knownHandsMask >> p) & 1U) != 0U;
    }

    [[nodiscard]] constexpr bool IsObservedMode() const noexcept {
        return knownHandsMask != 0x0FU;
    }

    [[nodiscard]] constexpr bool IsWaitingForDealInput() const noexcept {
        return (observedFlags & (kObservedFlagWaitingDeal | kObservedFlagWaitingRemSix)) != 0U;
    }

    [[nodiscard]] constexpr uint8_t ComputeVoidSuitsMask(const PlayerId p) const noexcept {
        uint8_t voids = 0;
        const HandMask imp = impossibleCards[p];
        for (uint8_t s = 0; s < 4U; ++s) {
            const HandMask sm = GetSuitMask(static_cast<Suit>(s));
            if ((imp & sm) == sm) {
                voids |= static_cast<uint8_t>(1U << s);
            }
        }
        return voids;
    }

    void ResetForNewMatch(RulePreset preset,
                          MatchMode matchMode = MatchMode::Long,
                          TeamMode tMode = TeamMode::Solo,
                          PulkaMode pMode = PulkaMode::Strict) noexcept;
    void ResetPulkaTrackers() noexcept;
    [[nodiscard]] GameStateView CreateView(PlayerId observer) const noexcept;
};

static_assert(sizeof(GameState) == 192, "GameState must fit in exactly three 64-byte cache lines");

} // namespace joker::core