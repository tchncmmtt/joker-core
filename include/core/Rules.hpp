#pragma once

#include "core/Action.hpp"
#include "core/GameState.hpp"

namespace joker::core::rules {

// 1. Размер колоды: 38 для Classic, 36 для Popular и Minus
constexpr size_t GetDeckSize(const RulePreset preset) noexcept {
    return (preset == RulePreset::Classic) ? kMaxDeckSize : kStandardDeckSize;
}

// Полная битовая маска колоды для заданного пресета
constexpr HandMask GetFullDeckMask(const RulePreset preset) noexcept {
    return (preset == RulePreset::Classic) ? kClassicDeckMask : kStandardDeckMask;
}

// 2. Выбор козыря по 3 картам на девятках: есть в Popular и Minus, нет в Classic
constexpr bool HasNinesTrumpBidding(const RulePreset preset) noexcept {
    return preset != RulePreset::Classic;
}

namespace detail {

constexpr ActionMask ComputeLegalActionsMask(
    const Phase phase,
    const int cardsInDeal,
    const PlayerId player,
    const PlayerId dealer,
    const std::array<int8_t, kNumPlayers>& bids,
    const HandMask hand,
    const int cardsInCurrentTrick,
    const std::array<TrickCard, kNumPlayers>& currentTrick,
    const Suit trumpSuit) noexcept {
    if (phase == Phase::ChoosingTrump) {
        return 0x1FULL; // Биты 0..4 (Spades, Clubs, Diamonds, Hearts, None)
    }

    if (phase == Phase::Bidding) {
        ActionMask mask = (1ULL << (cardsInDeal + 1)) - 1ULL;
        if (player == dealer) {
            int currentSum = 0;
            for (int p = 0; p < kNumPlayers; ++p) {
                if (p != player && bids[p] >= 0) {
                    currentSum += bids[p];
                }
            }
            if (const int forbiddenBid = cardsInDeal - currentSum;
                forbiddenBid >= 0 && forbiddenBid <= cardsInDeal) {
                mask &= ~(1ULL << forbiddenBid);
            }
        }
        return mask;
    }

    if (phase == Phase::TrickPlay) {
        const HandMask jokers = hand & kJokerMask;
        const HandMask nonJokers = hand & ~kJokerMask;
        const bool isLead = (cardsInCurrentTrick == 0);

        ActionMask mask = 0;
        if (jokers != 0) {
            mask |= isLead ? kJokerLeadActionsMask : kJokerFollowActionsMask;
        }

        if (isLead) {
            return mask | nonJokers;
        }

        const auto& [cardId, jokerAction, requestedSuit] = currentTrick[0].play;
        const Card& leadCard = GetCard(cardId);
        const Suit requiredSuit = leadCard.IsJoker() ? requestedSuit : leadCard.suit;
        const bool forceHighest =
            leadCard.IsJoker() && (jokerAction == JokerAction::LeadHighSuit);

        if (const HandMask suitCards = nonJokers & GetSuitMask(requiredSuit); suitCards != 0) {
            return mask | (forceHighest ? (1ULL << HighestCardInMask(suitCards)) : suitCards);
        }

        if (trumpSuit != Suit::None) {
            if (const HandMask trumpCards = nonJokers & GetSuitMask(trumpSuit); trumpCards != 0) {
                return mask | (forceHighest ? (1ULL << HighestCardInMask(trumpCards)) : trumpCards);
            }
        }

        return mask | nonJokers;
    }

    return 0ULL;
}

} // namespace detail

// 3. Генерация битовой маски легальных действий напрямую из GameState для заданного игрока
constexpr ActionMask GetLegalActionsMaskForState(const GameState& state, const PlayerId player) noexcept {
    return detail::ComputeLegalActionsMask(
        state.phase,
        state.cardsInDeal,
        player,
        state.dealer,
        state.bids,
        state.hands[player],
        state.cardsInCurrentTrick,
        state.currentTrick,
        state.trumpSuit);
}

// Совместимая перегрузка для GameStateView
constexpr ActionMask GetLegalActionsMask(const GameStateView& state) noexcept {
    return detail::ComputeLegalActionsMask(
        state.phase,
        state.cardsInDeal,
        state.observerPlayer,
        state.dealerPlayer,
        state.bids,
        state.myHand,
        state.cardsInCurrentTrick,
        state.currentTrick,
        state.trumpSuit);
}

// 4. Определение победителя взятки
constexpr PlayerId DetermineTrickWinner(
    const std::array<TrickCard, kNumPlayers>& trick,
    const Suit trumpSuit) noexcept {
    // Если кто-то сыграл Джокера на победу (PlayWin) — взятку забирает последний
    for (int i = kNumPlayers - 1; i >= 0; --i) {
        const auto& play = trick[i].play;
        if (GetCard(play.cardId).IsJoker() && play.jokerAction == JokerAction::PlayWin) {
            return trick[i].player;
        }
    }

    const auto& [cardId, jokerAction, requestedSuit] = trick[0].play;
    const Card& leadCard = GetCard(cardId);

    // Джокер ходил первым "По высшей" (LeadHighSuit)
    if (leadCard.IsJoker() && jokerAction == JokerAction::LeadHighSuit) {
        PlayerId winner = trick[0].player;
        auto bestTrumpRank = Rank::Six;

        if (trumpSuit != Suit::None && requestedSuit != trumpSuit) {
            bool trumped = false;
            for (int i = 1; i < kNumPlayers; ++i) {
                if (const Card& c = GetCard(trick[i].play.cardId); !c.IsJoker() && c.suit == trumpSuit) {
                    if (!trumped || c.rank > bestTrumpRank) {
                        trumped = true;
                        bestTrumpRank = c.rank;
                        winner = trick[i].player;
                    }
                }
            }
        }
        return winner;
    }

    // Обычный ход или Джокер "По меньшей" (LeadLow)
    const Suit effectiveLeadSuit = leadCard.IsJoker() ? requestedSuit : leadCard.suit;

    PlayerId winner = trick[0].player;
    Suit winningSuit = leadCard.IsJoker() ? Suit::None : leadCard.suit;
    Rank winningRank = leadCard.rank;
    bool hasValidCardWinner = !leadCard.IsJoker();

    for (int i = 1; i < kNumPlayers; ++i) {
        const CardPlay& play = trick[i].play;
        const Card& c = GetCard(play.cardId);
        if (c.IsJoker()) continue; // Джокер с PlayLow не берет

        if (!hasValidCardWinner) {
            if (c.suit == effectiveLeadSuit || (trumpSuit != Suit::None && c.suit == trumpSuit)) {
                hasValidCardWinner = true;
                winner = trick[i].player;
                winningSuit = c.suit;
                winningRank = c.rank;
            }
            continue;
        }

        if (c.suit == winningSuit) {
            if (c.rank > winningRank) {
                winningRank = c.rank;
                winner = trick[i].player;
            }
        } else if (trumpSuit != Suit::None && c.suit == trumpSuit && winningSuit != trumpSuit) {
            winningSuit = trumpSuit;
            winningRank = c.rank;
            winner = trick[i].player;
        }
    }

    return winner;
}

// 5. Подсчет очков
constexpr int CalculateDealScore(const RulePreset preset,
    const int bid, const int tricksTaken, const int numCards) noexcept {
    if (bid == 0 && tricksTaken == 0) {
        return 50;
    }

    if (tricksTaken == bid) {
        if (preset == RulePreset::Minus) {
            return 100 * tricksTaken;
        }
        if (preset == RulePreset::Popular && bid == numCards) {
            return 100 * tricksTaken;
        }
        return 50 + 50 * tricksTaken;
    }

    if (tricksTaken > bid) {
        return 10 * tricksTaken;
    }

    if (preset == RulePreset::Minus) {
        return -100 * (bid - tricksTaken);
    }

    if (tricksTaken == 0) {
        return (preset == RulePreset::Popular) ? -200 : 0;
    }

    return 10 * tricksTaken;
}

} // namespace joker::core::rules