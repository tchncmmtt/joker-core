#include "engine/JokerEngine.hpp"
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
        return false; // Защита от нелегального хода
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

        // Обновление жёсткой логической памяти (impossibleCards) при ответе обычной картой
        if (state_.cardsInCurrentTrick > 0 && !playedCard.IsJoker()) {
            const CardPlay& leadPlay = state_.currentTrick[0].play;
            const Card& leadCard = GetCard(leadPlay.cardId);
            const Suit reqSuit = leadCard.IsJoker() ? leadPlay.requestedSuit : leadCard.suit;
            const bool leadHigh = leadCard.IsJoker() && (leadPlay.jokerAction == JokerAction::LeadHighSuit);

            if (reqSuit != Suit::None) {
                if (playedCard.suit != reqSuit) {
                    // 1. Ренонс по требуемой масти
                    state_.impossibleCards[p] |= GetSuitMask(reqSuit);
                    if (state_.trumpSuit != Suit::None && reqSuit != state_.trumpSuit) {
                        if (playedCard.suit != state_.trumpSuit) {
                            // Нет ни требуемой масти, ни козыря
                            state_.impossibleCards[p] |= GetSuitMask(state_.trumpSuit);
                        } else if (leadHigh) {
                            // Ход по высшей, игрок бил козырем -> обязан был дать старший козырь
                            state_.impossibleCards[p] |=
                                GetHigherCardsInSuitMask(state_.trumpSuit, playedCard.rank);
                        }
                    }
                } else if (leadHigh) {
                    // 2. Ответ в требуемую масть на Джокера "по высшей" -> нет карт строго старше
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
    if (IsDone() || state_.IsWaitingForDealInput()) return;
    observer = static_cast<PlayerId>(observer % kNumPlayers);

    deck_.Reseed(rngSeed);

    const HandMask fullDeck = rules::GetFullDeckMask(state_.rules);
    const HandMask upcardMask = (state_.upcard != kInvalidCard && state_.upcard < kMaxDeckSize)
        ? (1ULL << state_.upcard)
        : 0ULL;
    const HandMask knownMask = state_.hands[observer] | state_.playedCardsHistory | upcardMask;
    const HandMask unknownMask = fullDeck & ~knownMask;

    std::array<CardId, kMaxDeckSize> pool{};
    uint32_t poolSize = 0;
    HandMask tmp = unknownMask;
    while (tmp != 0ULL) {
        const CardId cid = LowestCardInMask(tmp);
        tmp &= (tmp - 1ULL);
        pool[poolSize++] = cid;
    }

    if (poolSize == 0) return;

    std::array<PlayerId, 3> opps{};
    std::array<uint32_t, 3> need{};
    std::array<HandMask, 3> forbidden{};
    int oppIdx = 0;
    uint32_t totalNeeded = 0;

    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == observer) continue;
        opps[oppIdx] = p;
        need[oppIdx] = state_.handCounts[p];
        totalNeeded += need[oppIdx];
        forbidden[oppIdx] = state_.impossibleCards[p];
        oppIdx++;
    }

    if (totalNeeded > poolSize) return;

    for (uint32_t i = poolSize - 1U; i > 0U; --i) {
        const uint32_t j = deck_.NextBounded(i + 1U);
        std::swap(pool[i], pool[j]);
    }

    const std::array<uint32_t, 4> segStart{0, need[0], need[0] + need[1], totalNeeded};
    const std::array<uint32_t, 4> segEnd{need[0], need[0] + need[1], totalNeeded, poolSize};

    auto isCardAllowedForBucket = [&](const CardId c, const int bucket) noexcept -> bool {
        if (bucket >= 3) return true;
        return ((1ULL << c) & forbidden[bucket]) == 0ULL;
    };

    // Проход 1: быстрый жадный обмен вперёд
    for (int b = 0; b < 3; ++b) {
        for (uint32_t pos = segStart[b]; pos < segEnd[b]; ++pos) {
            if (!isCardAllowedForBucket(pool[pos], b)) {
                for (uint32_t cand = pos + 1U; cand < poolSize; ++cand) {
                    if (isCardAllowedForBucket(pool[cand], b)) {
                        std::swap(pool[pos], pool[cand]);
                        break;
                    }
                }
            }
        }
    }

    // Проход 2: парный обмен между любыми двумя корзинами
    for (int b = 0; b < 3; ++b) {
        for (uint32_t pos = segStart[b]; pos < segEnd[b]; ++pos) {
            const CardId badCard = pool[pos];
            if (isCardAllowedForBucket(badCard, b)) continue;

            bool repaired = false;
            for (int b2 = 3; b2 >= 0 && !repaired; --b2) {
                if (b2 == b) continue;
                for (uint32_t pos2 = segStart[b2]; pos2 < segEnd[b2]; ++pos2) {
                    const CardId otherCard = pool[pos2];
                    if (isCardAllowedForBucket(badCard, b2) && isCardAllowedForBucket(otherCard, b)) {
                        std::swap(pool[pos], pool[pos2]);
                        repaired = true;
                        break;
                    }
                }
            }
        }
    }

    // Проход 3: точный BFS по дополняющим циклам среди 4 корзин (0..3) на стеке, без аллокаций.
    // Гарантирует удовлетворение всех ограничений impossibleCards при наличии допустимого расклада.
    for (int b = 0; b < 3; ++b) {
        for (uint32_t pos = segStart[b]; pos < segEnd[b]; ++pos) {
            if (isCardAllowedForBucket(pool[pos], b)) continue;

            // Ищем кратчайший путь по корзинам: b -> u1 -> ... -> uk -> b
            std::array<int, 4> parentBucket{-1, -1, -1, -1};
            std::array<uint32_t, 4> edgePos{0, 0, 0, 0};
            std::array<int, 4> queue{};
            int qHead = 0;
            int qTail = 0;

            for (int nextB = 0; nextB < 4; ++nextB) {
                if (nextB == b || segStart[nextB] == segEnd[nextB]) continue;
                if (isCardAllowedForBucket(pool[pos], nextB)) {
                    parentBucket[nextB] = b;
                    edgePos[nextB] = pos;
                    queue[qTail++] = nextB;
                }
            }

            int foundEndBucket = -1;
            uint32_t closingPos = 0;

            while (qHead < qTail && foundEndBucket == -1) {
                const int curB = queue[qHead++];
                // Сначала проверяем, можно ли замкнуть цикл напрямую обратно в b
                for (uint32_t p2 = segStart[curB]; p2 < segEnd[curB]; ++p2) {
                    if (isCardAllowedForBucket(pool[p2], b)) {
                        foundEndBucket = curB;
                        closingPos = p2;
                        break;
                    }
                }
                if (foundEndBucket != -1) break;

                // Расширяем BFS в непосещённые корзины
                for (int nextB = 0; nextB < 4; ++nextB) {
                    if (nextB == b || parentBucket[nextB] != -1 || segStart[nextB] == segEnd[nextB]) {
                        continue;
                    }
                    for (uint32_t p2 = segStart[curB]; p2 < segEnd[curB]; ++p2) {
                        if (isCardAllowedForBucket(pool[p2], nextB)) {
                            parentBucket[nextB] = curB;
                            edgePos[nextB] = p2;
                            queue[qTail++] = nextB;
                            break;
                        }
                    }
                }
            }

            if (foundEndBucket != -1) {
                // Восстанавливаем цикл позиций и выполняем циклический сдвиг карт
                std::array<uint32_t, 4> cyclePos{};
                int len = 0;
                cyclePos[len++] = closingPos;
                int cur = foundEndBucket;
                while (cur != b && len < 4) {
                    cyclePos[len++] = edgePos[cur];
                    cur = parentBucket[cur];
                }
                // cyclePos[len-1] == pos (карта из b), cyclePos[0] == closingPos (карта для b)
                const CardId lastCard = pool[cyclePos[0]];
                for (int k = 0; k + 1 < len; ++k) {
                    pool[cyclePos[k]] = pool[cyclePos[k + 1]];
                }
                pool[cyclePos[len - 1]] = lastCard;
            }
        }
    }

    for (int b = 0; b < 3; ++b) {
        HandMask newHand = 0ULL;
        for (uint32_t pos = segStart[b]; pos < segEnd[b]; ++pos) {
            newHand |= (1ULL << pool[pos]);
        }
        state_.hands[opps[b]] = newHand;
    }

    // После детерминизации все 4 руки конкретизированы (в т.ч. при форке из Observed Mode)
    state_.knownHandsMask = 0x0F;

    if (state_.phase == Phase::ChoosingTrump) {
        HandMask remaining24Mask = 0ULL;
        for (uint32_t pos = segStart[3]; pos < segEnd[3]; ++pos) {
            remaining24Mask |= (1ULL << pool[pos]);
        }
        deck_.SetRemainingNinesStock(remaining24Mask);
    }
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
    // В мягком режиме сосед "над вами" (over = (p + 3) % 4, справа, ходит перед вами)
    // всегда является соперником как в Solo, так и в Pairs (P0+P2 против P1+P3).
    // 1. Вышедший игрок p получает премию, если не вышел игрок над ним (over).
    // 2. Невышедший игрок p получает списание, только если вышел игрок над ним (over).
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