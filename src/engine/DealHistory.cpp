#include "engine/DealHistory.hpp"
#include "core/Rules.hpp"
#include <algorithm>
#include <utility>

namespace joker::engine {

using namespace joker::core;

namespace {

constexpr uint8_t RelSeat1Based(const PlayerId absSeat, const PlayerId observer) noexcept {
    return static_cast<uint8_t>(
        ((static_cast<int>(absSeat) - static_cast<int>(observer) + kNumPlayers) % kNumPlayers) + 1);
}

} // namespace

DealHistoryBuffer::RawEvent DealHistoryBuffer::MakeEvent(
    const GameState& stBefore,
    const ActionId action) noexcept {
    RawEvent ev{};
    ev.absPlayer = stBefore.currentPlayer;
    ev.actionId = action;

    if (stBefore.phase == Phase::ChoosingTrump) {
        ev.eventType = 1U;
        ev.suit = action; // 0..4
    } else if (stBefore.phase == Phase::Bidding) {
        ev.eventType = 2U;
        ev.suit = 255U;
    } else if (stBefore.phase == Phase::TrickPlay) {
        ev.eventType = 3U;
        const PlayerId actor = stBefore.currentPlayer;
        const HandMask availMask = stBefore.IsHandKnown(actor)
            ? stBefore.hands[actor]
            : rules::ComputePossibleCardsForUnknownHand(stBefore, actor);
        const CardPlay play = DecodeCardPlayAction(action, availMask);
        ev.cardId = play.cardId;
        ev.jokerAction = std::to_underlying(play.jokerAction);
        const Card& c = GetCard(play.cardId);
        ev.suit = c.IsJoker() ? std::to_underlying(play.requestedSuit) : std::to_underlying(c.suit);
        ev.trickIdx = static_cast<uint8_t>(stBefore.tricksCompletedInDeal);
        ev.posInTrick = stBefore.cardsInCurrentTrick;
    }

    return ev;
}

void DealHistoryBuffer::WriteTokens(
    const PlayerId observer,
    int16_t* outTokens,
    uint8_t* outLength) const noexcept {
    constexpr size_t kTotalInts = kMaxSeqLen * kTokenDim;
    std::fill_n(outTokens, kTotalInts, static_cast<int16_t>(0));

    for (uint8_t i = 0; i < count; ++i) {
        const RawEvent& ev = events[i];
        int16_t* dst = outTokens + static_cast<size_t>(i) * kTokenDim;

        dst[0] = static_cast<int16_t>(ev.eventType); // 1..3 (0 = PAD)
        dst[1] = static_cast<int16_t>(RelSeat1Based(ev.absPlayer, observer)); // 1..4
        dst[2] = static_cast<int16_t>(ev.actionId + 1U); // 1..46
        dst[3] = (ev.cardId < kMaxDeckSize)
            ? static_cast<int16_t>(ev.cardId + 1U)
            : static_cast<int16_t>(0); // 1..38 или 0
        dst[4] = (ev.suit <= 4U)
            ? static_cast<int16_t>(ev.suit + 1U)
            : static_cast<int16_t>(0); // 1..5 или 0
        dst[5] = (ev.eventType == 3U)
            ? static_cast<int16_t>(ev.jokerAction + 1U)
            : static_cast<int16_t>(0); // 1..5 или 0
        dst[6] = (ev.trickIdx < 9U)
            ? static_cast<int16_t>(ev.trickIdx + 1U)
            : static_cast<int16_t>(0); // 1..9 или 0
        dst[7] = (ev.posInTrick < 4U)
            ? static_cast<int16_t>(ev.posInTrick + 1U)
            : static_cast<int16_t>(0); // 1..4 или 0
    }

    if (outLength) {
        *outLength = count;
    }
}

} // namespace joker::engine