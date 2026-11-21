#include "engine/JokerEngine.hpp"
#include "engine/Determinizer.hpp"
#include <algorithm>
#include <bit>
#include <utility>

namespace joker::engine {

using namespace joker::core;

JokerEngine::JokerEngine(RulePreset preset, uint64_t seed, MatchMode mode, TeamMode teamMode, PulkaMode pulkaMode) noexcept {
    Reset(preset, seed, mode, teamMode, pulkaMode);
}

void JokerEngine::Reset(RulePreset preset, uint64_t seed, MatchMode mode, TeamMode teamMode, PulkaMode pulkaMode) noexcept {
    deck_.Configure(seed, rules::GetDeckSize(preset));
    state_.ResetForNewMatch(preset, mode, teamMode, pulkaMode);
    BeginDeal(0);
}

void JokerEngine::Reset256(RulePreset preset, std::span<const uint64_t, 4> seed, MatchMode mode, TeamMode teamMode, PulkaMode pulkaMode) noexcept {
    deck_.Configure256(seed, rules::GetDeckSize(preset));
    state_.ResetForNewMatch(preset, mode, teamMode, pulkaMode);
    BeginDeal(0);
}

void JokerEngine::BeginObservedMatch(
    RulePreset preset,
    MatchMode mode,
    TeamMode teamMode,
    PulkaMode pulkaMode,
    PlayerId dealer,
    PlayerId heroSeat) noexcept {
    deck_.Configure(42ULL, rules::GetDeckSize(preset));
    state_.ResetForNewMatch(preset, mode, teamMode, pulkaMode);
    state_.dealer = static_cast<PlayerId>(dealer % kNumPlayers);
    state_.heroSeat = static_cast<PlayerId>(heroSeat % kNumPlayers);
    state_.knownHandsMask = static_cast<uint8_t>(1U << state_.heroSeat);
    BeginDeal(0);
}

bool JokerEngine::SetObservedMatchState(
        const RulePreset preset,
        const MatchMode mode,
        const TeamMode teamMode,
        const PulkaMode pulkaMode,
        const int dealNumber,
        const PlayerId dealer,
        const PlayerId heroSeat,
        const std::span<const int32_t, kNumPlayers> totalScores,
        const std::span<const uint8_t, kNumPlayers> pulkaClean,
        const std::span<const uint8_t, kNumPlayers> pulkaAllPasses,
        const std::span<const int16_t, kNumPlayers> pulkaMaxScoreExclLast,
        const std::span<const int16_t, kNumPlayers> pulkaMaxNonPassScoreExclLast) noexcept {
    if (dealNumber < 0 || dealNumber >= GetTotalDeals(mode)) {
        return false;
    }

    deck_.Configure(42ULL, rules::GetDeckSize(preset));
    state_.ResetForNewMatch(preset, mode, teamMode, pulkaMode);
    state_.dealer = static_cast<PlayerId>(dealer % kNumPlayers);
    state_.heroSeat = static_cast<PlayerId>(heroSeat % kNumPlayers);
    state_.knownHandsMask = static_cast<uint8_t>(1U << state_.heroSeat);

    BeginDeal(dealNumber);

    for (int p = 0; p < kNumPlayers; ++p) {
        state_.totalScores[p] = totalScores[p];
    }

    // Если вход не на первой сдаче пульки — восстанавливаем трекеры текущей пульки
    if (GetScheduleEntry(mode, dealNumber).roundInPulka > 0) {
        for (int p = 0; p < kNumPlayers; ++p) {
            state_.pulkaClean[p] = (pulkaClean[p] != 0U);
            state_.pulkaAllPasses[p] = (pulkaAllPasses[p] != 0U);
            state_.pulkaMaxScoreExclLast[p] = pulkaMaxScoreExclLast[p];
            state_.pulkaMaxNonPassScoreExclLast[p] = pulkaMaxNonPassScoreExclLast[p];
        }
    }

    return true;
}

bool JokerEngine::SetObservedDeal(
    const HandMask heroHandMask,
    const CardId upcardOrInvalid,
    const uint8_t trumpSuitOrNone) noexcept {
    if (IsDone() || !state_.IsObservedMode()) {
        return false;
    }
    if ((state_.observedFlags & GameState::kObservedFlagWaitingDeal) == 0U) {
        return false;
    }

    const HandMask fullDeck = rules::GetFullDeckMask(state_.rules);
    if ((heroHandMask & ~fullDeck) != 0ULL) {
        return false;
    }

    const auto& sched = GetScheduleEntry(state_.mode, state_.dealNumber);
    const bool isNinesGuess = sched.isNineCardDeal && rules::HasNinesTrumpBidding(state_.rules);
    const int expectedHeroCards = isNinesGuess ? 3 : sched.cardsPerPlayer;

    if (std::popcount(heroHandMask) != expectedHeroCards) {
        return false;
    }

    if (isNinesGuess) {
        if (upcardOrInvalid != kInvalidCard) {
            return false;
        }
        state_.upcard = kInvalidCard;
        state_.trumpSuit = Suit::None;
        state_.phase = Phase::ChoosingTrump;
    } else {
        Suit resolvedTrump = Suit::None;
        if (upcardOrInvalid != kInvalidCard) {
            if (upcardOrInvalid >= kMaxDeckSize) {
                return false;
            }
            const HandMask upBit = 1ULL << upcardOrInvalid;
            if ((fullDeck & upBit) == 0ULL || (heroHandMask & upBit) != 0ULL) {
                return false;
            }
            const Card& upCard = GetCard(upcardOrInvalid);
            const Suit derivedTrump = upCard.IsJoker() ? Suit::None : upCard.suit;
            if (trumpSuitOrNone != 255U) {
                if (trumpSuitOrNone > 4U || static_cast<Suit>(trumpSuitOrNone) != derivedTrump) {
                    return false;
                }
            }
            state_.upcard = upcardOrInvalid;
            resolvedTrump = derivedTrump;
        } else {
            if (trumpSuitOrNone > 4U) {
                return false;
            }
            state_.upcard = kInvalidCard;
            resolvedTrump = static_cast<Suit>(trumpSuitOrNone);
        }
        state_.trumpSuit = resolvedTrump;
        state_.phase = Phase::Bidding;
    }

    state_.hands.fill(0ULL);
    state_.hands[state_.heroSeat] = heroHandMask;
    state_.handCounts.fill(static_cast<uint8_t>(expectedHeroCards));
    state_.observedFlags = 0U;

    const auto firstPlayer = static_cast<PlayerId>((state_.dealer + 1) % kNumPlayers);
    state_.trickLeader = firstPlayer;
    state_.currentPlayer = firstPlayer;
    return true;
}

bool JokerEngine::SetObservedRemainingSix(const HandMask fullNineHandMask) noexcept {
    if (IsDone() || !state_.IsObservedMode()) {
        return false;
    }
    if ((state_.observedFlags & GameState::kObservedFlagWaitingRemSix) == 0U) {
        return false;
    }

    const HandMask fullDeck = rules::GetFullDeckMask(state_.rules);
    if ((fullNineHandMask & ~fullDeck) != 0ULL) {
        return false;
    }
    if (std::popcount(fullNineHandMask) != 9) {
        return false;
    }
    const HandMask prevThree = state_.hands[state_.heroSeat];
    if ((fullNineHandMask & prevThree) != prevThree) {
        return false;
    }

    state_.hands[state_.heroSeat] = fullNineHandMask;
    state_.handCounts.fill(9U);
    state_.observedFlags = 0U;
    state_.phase = Phase::Bidding;
    state_.currentPlayer = static_cast<PlayerId>((state_.dealer + 1) % kNumPlayers);
    return true;
}

ActionMask JokerEngine::GetLegalActionsMask() const noexcept {
    if (IsDone()) return 0ULL;
    return rules::GetLegalActionsMaskForState(state_, state_.currentPlayer);
}

StepError JokerEngine::StepChecked(const ActionId action) noexcept {
    const StepError err = ValidateAction(action);
    if (err != StepError::Ok) {
        return err;
    }
    Step(action);
    return StepError::Ok;
}

bool JokerEngine::Step(ActionId action) noexcept {
    if (const ActionMask legalMask = GetLegalActionsMask(); action >= 64 || ((legalMask >> action) & 1ULL) == 0ULL) {
        return false;
    }

    const PlayerId p = state_.currentPlayer;

    if (state_.phase == Phase::ChoosingTrump) {
        state_.trumpSuit = static_cast<Suit>(action);
        if (state_.IsObservedMode()) {
            state_.observedFlags |= GameState::kObservedFlagWaitingRemSix;
        } else {
            deck_.DealRemainingSix(state_.hands);
            state_.handCounts.fill(9U);
        }
        state_.phase = Phase::Bidding;
        state_.currentPlayer = static_cast<PlayerId>((state_.dealer + 1) % kNumPlayers);
        return true;
    }

    if (state_.phase == Phase::Bidding) {
        state_.bids[p] = static_cast<int8_t>(action);
        state_.bidsMadeInDeal++;

        if (state_.bidsMadeInDeal >= kNumPlayers) {
            state_.phase = Phase::TrickPlay;
            state_.trickLeader = static_cast<PlayerId>((state_.dealer + 1) % kNumPlayers);
            state_.currentPlayer = state_.trickLeader;
            state_.cardsInCurrentTrick = 0;
        } else {
            state_.currentPlayer = static_cast<PlayerId>((p + 1) % kNumPlayers);
        }
        return true;
    }

    if (state_.phase == Phase::TrickPlay) {
        const HandMask availableMask = state_.IsHandKnown(p)
            ? state_.hands[p]
            : rules::ComputePossibleCardsForUnknownHand(state_, p);
        const CardPlay move = DecodeCardPlayAction(action, availableMask);
        const Card& playedCard = GetCard(move.cardId);

        if (state_.cardsInCurrentTrick > 0 && !playedCard.IsJoker()) {
            const CardPlay& leadPlay = state_.currentTrick[0].play;
            const Card& leadCard = GetCard(leadPlay.cardId);
            const Suit reqSuit = leadCard.IsJoker() ? leadPlay.requestedSuit : leadCard.suit;
            const bool leadHigh = leadCard.IsJoker() && (leadPlay.jokerAction == JokerAction::LeadHighSuit);

            if (reqSuit != Suit::None) {
                if (playedCard.suit != reqSuit) {
                    state_.impossibleCards[p] |= GetSuitMask(reqSuit);
                    if (state_.trumpSuit != Suit::None && reqSuit != state_.trumpSuit) {
                        if (playedCard.suit != state_.trumpSuit) {
                            state_.impossibleCards[p] |= GetSuitMask(state_.trumpSuit);
                        } else if (leadHigh) {
                            state_.impossibleCards[p] |=
                                GetHigherCardsInSuitMask(state_.trumpSuit, playedCard.rank);
                        }
                    }
                } else if (leadHigh) {
                    state_.impossibleCards[p] |= GetHigherCardsInSuitMask(reqSuit, playedCard.rank);
                }
            }
        }

        if (playedCard.IsJoker()) {
            state_.jokersPlayedBy[p]++;
        }

        if (state_.IsHandKnown(p)) {
            state_.hands[p] &= ~(1ULL << move.cardId);
        }
        if (state_.handCounts[p] > 0U) {
            state_.handCounts[p]--;
        }
        state_.playedCardsHistory |= (1ULL << move.cardId);
        state_.currentTrick[state_.cardsInCurrentTrick++] = TrickCard{p, move};

        if (state_.cardsInCurrentTrick == kNumPlayers) {
            const PlayerId winner = rules::DetermineTrickWinner(state_.currentTrick, state_.trumpSuit);
            state_.lastTrickWinner = winner;
            state_.tricksTaken[winner]++;
            state_.tricksCompletedInDeal++;

            if (state_.tricksCompletedInDeal >= state_.cardsInDeal) {
                FinalizeDealScoring();
                state_.dealer = static_cast<PlayerId>((state_.dealer + 1) % kNumPlayers);
                BeginDeal(state_.dealNumber + 1);
            } else {
                state_.trickLeader = winner;
                state_.currentPlayer = winner;
                state_.cardsInCurrentTrick = 0;
            }
        } else {
            state_.currentPlayer = static_cast<PlayerId>((p + 1) % kNumPlayers);
        }
        return true;
    }

    return false;
}

void JokerEngine::Determinize(PlayerId observer, const uint64_t rngSeed) noexcept {
    DeterminizeState(state_, deck_, observer, rngSeed);
}

void JokerEngine::BeginDeal(const int dealIdx) noexcept {
    if (dealIdx >= GetTotalDeals(state_.mode)) {
        state_.phase = Phase::MatchFinished;
        state_.observedFlags = 0U;
        return;
    }

    const auto& sched = GetScheduleEntry(state_.mode, dealIdx);
    if (sched.roundInPulka == 0) {
        state_.ResetPulkaTrackers();
    }

    state_.dealNumber = static_cast<int8_t>(dealIdx);
    state_.cardsInDeal = static_cast<int8_t>(sched.cardsPerPlayer);
    state_.playedCardsHistory = 0ULL;
    state_.bids.fill(-1);
    state_.tricksTaken.fill(0);
    state_.impossibleCards.fill(0ULL);
    state_.jokersPlayedBy.fill(0);
    state_.cardsInCurrentTrick = 0;
    state_.bidsMadeInDeal = 0;
    state_.tricksCompletedInDeal = 0;

    const auto firstPlayer = static_cast<PlayerId>((state_.dealer + 1) % kNumPlayers);
    state_.trickLeader = firstPlayer;
    state_.currentPlayer = firstPlayer;

    if (state_.IsObservedMode()) {
        state_.hands.fill(0ULL);
        state_.handCounts.fill(0U);
        state_.upcard = kInvalidCard;
        state_.trumpSuit = Suit::None;
        state_.phase = (sched.isNineCardDeal && rules::HasNinesTrumpBidding(state_.rules))
            ? Phase::ChoosingTrump
            : Phase::Bidding;
        state_.observedFlags = GameState::kObservedFlagWaitingDeal;
        return;
    }

    state_.observedFlags = 0U;
    if (sched.isNineCardDeal && rules::HasNinesTrumpBidding(state_.rules)) {
        state_.hands = deck_.DealFirstThree();
        state_.handCounts.fill(3U);
        state_.upcard = kInvalidCard;
        state_.trumpSuit = Suit::None;
        state_.phase = Phase::ChoosingTrump;
    } else {
        const DealResult d = deck_.DealStandard(sched.cardsPerPlayer);
        state_.hands = d.hands;
        state_.handCounts.fill(static_cast<uint8_t>(sched.cardsPerPlayer));
        state_.upcard = d.upcard;
        state_.trumpSuit = d.trumpSuit;
        state_.phase = Phase::Bidding;
    }
}

void JokerEngine::FinalizeDealScoring() noexcept {
    const auto& sched = GetScheduleEntry(state_.mode, state_.dealNumber);

    for (int p = 0; p < kNumPlayers; ++p) {
        const int b = state_.bids[p];
        const int t = state_.tricksTaken[p];
        const int score = rules::CalculateDealScore(state_.rules, b, t, state_.cardsInDeal);

        state_.totalScores[p] += score;

        const bool exact = (b == t);
        if (!exact) {
            state_.pulkaClean[p] = false;
        }
        if (!exact || b != 0) {
            state_.pulkaAllPasses[p] = false;
        }
        if (!sched.isPulkaEnd) {
            if (score > state_.pulkaMaxScoreExclLast[p]) {
                state_.pulkaMaxScoreExclLast[p] = static_cast<int16_t>(score);
            }
            const bool isPass = (b == 0 && t == 0);
            if (!isPass && score > state_.pulkaMaxNonPassScoreExclLast[p]) {
                state_.pulkaMaxNonPassScoreExclLast[p] = static_cast<int16_t>(score);
            }
        }
    }

    if (sched.isPulkaEnd) {
        ApplyPulkaBonuses();
    }
}

void JokerEngine::ApplyPulkaBonuses() noexcept {
    if (state_.pulkaMode == PulkaMode::Soft) {
        ApplySoftPulkaBonuses();
    } else if (state_.teamMode == TeamMode::Pairs) {
        ApplyPairsStrictPulkaBonuses();
    } else {
        ApplySoloStrictPulkaBonuses();
    }
}

void JokerEngine::ApplySoloStrictPulkaBonuses() noexcept {
    int numQualified = 0;
    for (int p = 0; p < kNumPlayers; ++p) {
        if (state_.pulkaClean[p]) {
            numQualified++;
        }
    }

    if (numQualified == 0) {
        return;
    }

    for (int p = 0; p < kNumPlayers; ++p) {
        int bonus = 0;
        if (state_.pulkaClean[p]) {
            if (numQualified > 1) {
                bonus = 0;
            } else if (state_.pulkaAllPasses[p]) {
                bonus = 450;
            } else {
                bonus = state_.pulkaMaxScoreExclLast[p];
            }
        } else {
            bonus = -static_cast<int>(state_.pulkaMaxNonPassScoreExclLast[p]);
        }
        state_.totalScores[p] += bonus;
    }
}

void JokerEngine::ApplyPairsStrictPulkaBonuses() noexcept {
    const bool teamAClean = state_.pulkaClean[0] || state_.pulkaClean[2];
    const bool teamBClean = state_.pulkaClean[1] || state_.pulkaClean[3];

    if (!teamAClean && !teamBClean) {
        return;
    }

    const bool bothTeamsQualified = teamAClean && teamBClean;

    for (int p = 0; p < kNumPlayers; ++p) {
        const bool isTeamA = (p == 0 || p == 2);
        const bool ownTeamClean = isTeamA ? teamAClean : teamBClean;
        int delta = 0;

        if (state_.pulkaClean[p]) {
            if (!bothTeamsQualified) {
                delta = state_.pulkaAllPasses[p] ? 450 : state_.pulkaMaxScoreExclLast[p];
            }
        } else if (!ownTeamClean) {
            delta = -static_cast<int>(state_.pulkaMaxNonPassScoreExclLast[p]);
        }

        state_.totalScores[p] += delta;
    }
}

void JokerEngine::ApplySoftPulkaBonuses() noexcept {
    for (int p = 0; p < kNumPlayers; ++p) {
        const int over = (p + kNumPlayers - 1) % kNumPlayers;
        int delta = 0;

        if (state_.pulkaClean[p]) {
            if (!state_.pulkaClean[over]) {
                delta = state_.pulkaAllPasses[p] ? 450 : state_.pulkaMaxScoreExclLast[p];
            }
        } else if (state_.pulkaClean[over]) {
            delta = -static_cast<int>(state_.pulkaMaxNonPassScoreExclLast[p]);
        }

        state_.totalScores[p] += delta;
    }
}

} // namespace joker::engine