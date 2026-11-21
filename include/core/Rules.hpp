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

// Вычисление пула потенциально возможных карт на руке игрока с неизвестной рукой (Observed Mode)
constexpr HandMask ComputePossibleCardsForUnknownHand(
    const GameState& state,
    const PlayerId player) noexcept {
    HandMask knownOtherHands = 0ULL;
    for (PlayerId q = 0; q < kNumPlayers; ++q) {
        if (q != player && state.IsHandKnown(q)) {
            knownOtherHands |= state.hands[q];
        }
    }
    const HandMask upcardMask = (state.upcard != kInvalidCard && state.upcard < kMaxDeckSize)
        ? (1ULL << state.upcard)
        : 0ULL;
    const HandMask excluded = state.playedCardsHistory | knownOtherHands | upcardMask | state.impossibleCards[player];
    return GetFullDeckMask(state.rules) & ~excluded;
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
    if (state.phase == Phase::MatchFinished || state.IsWaitingForDealInput()) {
        return 0ULL;
    }

    if (state.phase == Phase::TrickPlay && !state.IsHandKnown(player)) {
        const HandMask possible = ComputePossibleCardsForUnknownHand(state, player);
        const bool isLead = (state.cardsInCurrentTrick == 0);
        ActionMask mask = possible & ~kJokerMask;
        if ((possible & kJokerMask) != 0ULL) {
            mask |= isLead ? kJokerLeadActionsMask : kJokerFollowActionsMask;
        }
        return mask;
    }

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

// 4. Определение победителя взятки (поддерживает и полную взятку count==4, и частичную count==1..3)
constexpr PlayerId DetermineTrickWinnerN(
    const std::array<TrickCard, kNumPlayers>& trick,
    const uint8_t count,
    const Suit trumpSuit) noexcept {
    JOKER_ASSUME(count > 0 && count <= kNumPlayers);

    // Если кто-то сыграл Джокера на победу (PlayWin) — взятку забирает последний
    for (int i = static_cast<int>(count) - 1; i >= 0; --i) {
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
            for (uint8_t i = 1; i < count; ++i) {
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

    for (uint8_t i = 1; i < count; ++i) {
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

constexpr PlayerId DetermineTrickWinner(
    const std::array<TrickCard, kNumPlayers>& trick,
    const Suit trumpSuit) noexcept {
    return DetermineTrickWinnerN(trick, static_cast<uint8_t>(kNumPlayers), trumpSuit);
}

// 5. Подсчет очков за сдачу
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

// 6. Детальная валидация хода с кодом ошибки (для Live/Observed Mode и UI)
constexpr StepError ValidateActionForState(const GameState& state, const ActionId action) noexcept {
    if (state.phase == Phase::MatchFinished) {
        return StepError::MatchFinished;
    }
    if (state.IsWaitingForDealInput()) {
        return StepError::WaitingForDealInput;
    }

    const PlayerId p = state.currentPlayer;

    if (state.phase == Phase::ChoosingTrump) {
        return (action <= 4U) ? StepError::Ok : StepError::InvalidActionId;
    }

    if (state.phase == Phase::Bidding) {
        if (action > static_cast<ActionId>(state.cardsInDeal)) {
            return StepError::InvalidActionId;
        }
        if (p == state.dealer) {
            int currentSum = 0;
            for (int q = 0; q < kNumPlayers; ++q) {
                if (q != p && state.bids[q] >= 0) {
                    currentSum += state.bids[q];
                }
            }
            if (currentSum + static_cast<int>(action) == state.cardsInDeal) {
                return StepError::ForbiddenDealerBid;
            }
        }
        return StepError::Ok;
    }

    if (state.phase == Phase::TrickPlay) {
        if (action >= kMaxActions) {
            return StepError::InvalidActionId;
        }
        const bool isLead = (state.cardsInCurrentTrick == 0);
        const HandMask fullDeck = GetFullDeckMask(state.rules);
        const HandMask upcardMask = (state.upcard != kInvalidCard && state.upcard < kMaxDeckSize)
            ? (1ULL << state.upcard)
            : 0ULL;

        HandMask knownOtherHands = 0ULL;
        for (PlayerId q = 0; q < kNumPlayers; ++q) {
            if (q != p && state.IsHandKnown(q)) {
                knownOtherHands |= state.hands[q];
            }
        }

        // Действие Джокером
        if (action >= 36U) {
            if (isLead) {
                if (action < kActionJokerLeadHighBase) {
                    return StepError::InvalidActionId;
                }
            } else {
                if (action != kActionJokerPlayWin && action != kActionJokerPlayLow) {
                    return StepError::InvalidActionId;
                }
            }

            if (state.IsHandKnown(p)) {
                return ((state.hands[p] & kJokerMask) != 0ULL)
                    ? StepError::Ok
                    : StepError::NoJokerAvailable;
            }
            const HandMask availableJokers =
                kJokerMask & ~(state.playedCardsHistory | knownOtherHands | upcardMask | state.impossibleCards[p]);
            return (availableJokers != 0ULL) ? StepError::Ok : StepError::NoJokerAvailable;
        }

        // Действие обычной картой (0..35)
        const HandMask cardBit = 1ULL << action;
        if ((fullDeck & cardBit) == 0ULL) {
            return StepError::InvalidActionId;
        }
        if ((state.playedCardsHistory & cardBit) != 0ULL) {
            return StepError::CardAlreadyPlayed;
        }
        if ((upcardMask & cardBit) != 0ULL) {
            return StepError::CardIsUpcard;
        }
        if ((knownOtherHands & cardBit) != 0ULL) {
            return StepError::CardInOtherKnownHand;
        }

        if (state.IsHandKnown(p)) {
            if ((state.hands[p] & cardBit) == 0ULL) {
                return StepError::CardNotInHand;
            }
            if (!isLead) {
                const auto& [leadCardId, leadJokerAction, leadReqSuit] = state.currentTrick[0].play;
                const Card& leadCard = GetCard(leadCardId);
                const Suit requiredSuit = leadCard.IsJoker() ? leadReqSuit : leadCard.suit;
                const bool forceHighest =
                    leadCard.IsJoker() && (leadJokerAction == JokerAction::LeadHighSuit);

                const HandMask nonJokers = state.hands[p] & ~kJokerMask;
                if (const HandMask suitCards = nonJokers & GetSuitMask(requiredSuit); suitCards != 0ULL) {
                    if ((suitCards & cardBit) == 0ULL) {
                        return StepError::MustFollowSuit;
                    }
                    if (forceHighest && action != HighestCardInMask(suitCards)) {
                        return StepError::MustPlayHighest;
                    }
                } else if (state.trumpSuit != Suit::None) {
                    if (const HandMask trumpCards = nonJokers & GetSuitMask(state.trumpSuit); trumpCards != 0ULL) {
                        if ((trumpCards & cardBit) == 0ULL) {
                            return StepError::MustPlayTrump;
                        }
                        if (forceHighest && action != HighestCardInMask(trumpCards)) {
                            return StepError::MustPlayHighest;
                        }
                    }
                }
            }
            return StepError::Ok;
        }

        // Неизвестная рука (Observed Mode)
        if ((state.impossibleCards[p] & cardBit) != 0ULL) {
            return StepError::ViolatesKnownVoidOrBound;
        }
        return StepError::Ok;
    }

    return StepError::InvalidActionId;
}

} // namespace joker::core::rules