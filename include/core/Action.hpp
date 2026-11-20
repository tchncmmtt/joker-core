#pragma once

#include "core/Card.hpp"
#include <cstdint>

namespace joker::core {

enum class Phase : uint8_t {
    ChoosingTrump = 0, // ActionId: 0..4 (Suit::Spades ... Suit::None)
    Bidding       = 1, // ActionId: 0..9 (Заказ 0..9 взяток)
    TrickPlay     = 2, // ActionId: 0..45 (0..35 карты, 36..45 Джокер)
    MatchFinished = 3
};

using ActionId = uint8_t;
using ActionMask = uint64_t;

constexpr ActionId kInvalidAction = 255;

constexpr ActionId kActionJokerPlayWin      = 36;
constexpr ActionId kActionJokerPlayLow      = 37;
constexpr ActionId kActionJokerLeadHighBase = 38; // 38..41 (+ Suit 0..3)
constexpr ActionId kActionJokerLeadLowBase  = 42; // 42..45 (+ Suit 0..3)
constexpr size_t   kMaxActions              = 46;

// 8 действий первого хода Джокером (биты 38..45)
inline constexpr ActionMask kJokerLeadActionsMask   = ((1ULL << 8) - 1ULL) << 38;
// 2 действия ответа Джокером (биты 36..37)
inline constexpr ActionMask kJokerFollowActionsMask = (1ULL << 36) | (1ULL << 37);

constexpr ActionId EncodeCardPlayAction(const CardPlay& play) noexcept {
    if (!GetCard(play.cardId).IsJoker()) {
        return play.cardId;
    }
    switch (play.jokerAction) {
        case JokerAction::PlayWin:
            return kActionJokerPlayWin;
        case JokerAction::PlayLow:
            return kActionJokerPlayLow;
        case JokerAction::LeadHighSuit:
            return static_cast<ActionId>(kActionJokerLeadHighBase + static_cast<uint8_t>(play.requestedSuit));
        case JokerAction::LeadLow:
            return static_cast<ActionId>(kActionJokerLeadLowBase + static_cast<uint8_t>(play.requestedSuit));
        default:
            return kInvalidAction;
    }
}

constexpr CardPlay DecodeCardPlayAction(const ActionId action, const HandMask playerHand) noexcept {
    if (action < 36) {
        return CardPlay{.cardId = action, .jokerAction = JokerAction::None, .requestedSuit = Suit::None};
    }

    const CardId jokerId = ((playerHand & (1ULL << 36)) != 0) ? 36 : 37;

    if (action == kActionJokerPlayWin) {
        return CardPlay{.cardId = jokerId, .jokerAction = JokerAction::PlayWin, .requestedSuit = Suit::None};
    }
    if (action == kActionJokerPlayLow) {
        return CardPlay{.cardId = jokerId, .jokerAction = JokerAction::PlayLow, .requestedSuit = Suit::None};
    }
    if (action >= kActionJokerLeadHighBase && action < kActionJokerLeadHighBase + 4) {
        return CardPlay{.cardId = jokerId, .jokerAction = JokerAction::LeadHighSuit,
            .requestedSuit = static_cast<Suit>(action - kActionJokerLeadHighBase)};
    }
    if (action >= kActionJokerLeadLowBase && action < kActionJokerLeadLowBase + 4) {
        return CardPlay{.cardId = jokerId, .jokerAction = JokerAction::LeadLow,
            .requestedSuit = static_cast<Suit>(action - kActionJokerLeadLowBase)};
    }
    return CardPlay{};
}

} // namespace joker::core