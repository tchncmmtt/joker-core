#include "engine/HeuristicBot.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <utility>

namespace joker::engine {

using namespace joker::core;

namespace {

inline ActionId PickRandomLegalAction(ActionMask mask, uint64_t& rngState) noexcept {
    if (mask == 0ULL) return 0U;
    const uint32_t count = static_cast<uint32_t>(std::popcount(mask));
    if (count == 1U) {
        return static_cast<ActionId>(std::countr_zero(mask));
    }
    const uint32_t target = static_cast<uint32_t>(Deck::SplitMix64(rngState) % count);
    for (uint32_t k = 0; k < target; ++k) {
        mask &= (mask - 1ULL);
    }
    return static_cast<ActionId>(std::countr_zero(mask));
}

inline ActionId HeuristicChooseTrump(
    const GameState& st,
    const ActionMask legalMask,
    uint64_t& rngState) noexcept {
    const HandMask hand = st.hands[st.currentPlayer];
    const int nJokers = std::popcount(hand & kJokerMask);

    int numAces = 0;
    int numKings = 0;
    int maxSuitLen = 0;

    std::array<float, 5> scores{-1e9f, -1e9f, -1e9f, -1e9f, -1e9f};

    for (uint8_t s = 0; s < 4U; ++s) {
        const HandMask sCards = hand & GetSuitMask(static_cast<Suit>(s));
        const int cnt = std::popcount(sCards);
        if (cnt > maxSuitLen) maxSuitLen = cnt;

        float val = (cnt == 3) ? 4.3f : ((cnt == 2) ? 2.15f : ((cnt == 1) ? 0.45f : -2.0f));
        HandMask m = sCards;
        while (m != 0ULL) {
            const CardId cid = LowestCardInMask(m);
            m &= (m - 1ULL);
            const Rank r = GetCard(cid).rank;
            switch (r) {
                case Rank::Ace:   val += 2.2f; ++numAces; break;
                case Rank::King:  val += 1.3f; ++numKings; break;
                case Rank::Queen: val += 0.7f; break;
                case Rank::Jack:  val += 0.35f; break;
                case Rank::Ten:   val += 0.15f; break;
                default: break;
            }
        }
        if (nJokers > 0 && cnt >= 2) {
            val += 0.35f * static_cast<float>(nJokers);
        }
        scores[s] = val;
    }

    // Оценка бескозырки (Suit::None = 4)
    float ntScore = 1.65f * static_cast<float>(nJokers)
                  + 1.45f * static_cast<float>(numAces)
                  + 0.55f * static_cast<float>(numKings)
                  - 0.45f;
    if (nJokers >= 2) {
        ntScore += 1.4f;
    } else if (nJokers == 1 && maxSuitLen <= 1) {
        ntScore += 0.9f;
    }
    scores[4] = ntScore;

    ActionId bestAct = static_cast<ActionId>(std::countr_zero(legalMask));
    float bestVal = -1e9f;
    const uint64_t tie = Deck::SplitMix64(rngState);
    for (uint8_t a = 0; a < 5U; ++a) {
        if (((legalMask >> a) & 1ULL) == 0ULL) continue;
        const float jitter = static_cast<float>((tie >> (a * 8U)) & 0xFFU) * 1e-5f;
        if (scores[a] + jitter > bestVal) {
            bestVal = scores[a] + jitter;
            bestAct = a;
        }
    }
    return bestAct;
}

inline ActionId HeuristicBid(
    const GameState& st,
    const ActionMask legalMask,
    uint64_t& rngState) noexcept {
    const PlayerId p = st.currentPlayer;
    const int n = st.cardsInDeal;
    const HandMask hand = st.hands[p];
    const HandMask fullDeck = rules::GetFullDeckMask(st.rules);
    const HandMask upcardMask = (st.upcard != kInvalidCard && st.upcard < kMaxDeckSize)
        ? (1ULL << st.upcard)
        : 0ULL;
    const float deckSizeF = static_cast<float>(rules::GetDeckSize(st.rules));
    const float dealFrac = std::min(1.0f, static_cast<float>(4 * n) / deckSizeF);
    const Suit trump = st.trumpSuit;

    float est = 0.0f;

    // 1. Джокеры
    const int nJokers = std::popcount(hand & kJokerMask);
    const float jokerVal = (trump == Suit::None) ? 0.97f : 0.93f;
    est += static_cast<float>(nJokers) * jokerVal;

    // 2. Козыри
    int trumpLen = 0;
    if (trump != Suit::None) {
        const HandMask tCards = hand & GetSuitMask(trump);
        trumpLen = std::popcount(tCards);
        HandMask m = tCards;
        while (m != 0ULL) {
            const CardId cid = LowestCardInMask(m);
            m &= (m - 1ULL);
            const Rank r = GetCard(cid).rank;
            const int higherOut = std::popcount(GetHigherCardsInSuitMask(trump, r) & fullDeck & ~hand & ~upcardMask);
            const float effHigher = static_cast<float>(higherOut) * dealFrac;
            if (effHigher < 0.35f) {
                est += 0.96f;
            } else if (effHigher < 1.15f) {
                est += (trumpLen >= 2) ? 0.80f : 0.56f;
            } else if (effHigher < 2.15f) {
                est += (trumpLen >= 3) ? 0.62f : 0.32f;
            } else if (effHigher < 3.25f) {
                est += (trumpLen >= 4) ? 0.42f : 0.14f;
            } else if (trumpLen >= 4) {
                est += 0.20f;
            }
        }
        if (trumpLen >= 3 && n >= 4) {
            est += 0.22f * static_cast<float>(trumpLen - 2);
        }
    }

    // 3. Побочные масти (или все 4 при Б/К)
    int voidSideSuits = 0;
    const float trumpCutRisk = (trump == Suit::None) ? 0.0f : ((n >= 6) ? 0.22f : ((n >= 3) ? 0.14f : 0.08f));

    for (uint8_t s = 0; s < 4U; ++s) {
        const Suit suit = static_cast<Suit>(s);
        if (suit == trump) continue;
        const HandMask sCards = hand & GetSuitMask(suit);
        const int sLen = std::popcount(sCards);
        if (sLen == 0) {
            ++voidSideSuits;
            continue;
        }
        HandMask m = sCards;
        while (m != 0ULL) {
            const CardId cid = LowestCardInMask(m);
            m &= (m - 1ULL);
            const Rank r = GetCard(cid).rank;
            const int higherOut = std::popcount(GetHigherCardsInSuitMask(suit, r) & fullDeck & ~hand & ~upcardMask);
            const float effHigher = static_cast<float>(higherOut) * dealFrac;

            if (effHigher < 0.35f) {
                const float base = (trump == Suit::None) ? 0.92f : (0.86f * (1.0f - trumpCutRisk));
                est += base;
            } else if (effHigher < 1.15f) {
                float base = (sLen >= 2) ? 0.54f : ((n <= 2) ? 0.36f : 0.18f);
                if (trump != Suit::None) base *= (1.0f - trumpCutRisk);
                est += base;
            } else if (effHigher < 2.15f) {
                float base = (sLen >= 3) ? 0.24f : ((n == 1) ? 0.18f : 0.05f);
                if (trump != Suit::None) base *= (1.0f - trumpCutRisk);
                est += base;
            } else if (trump == Suit::None && sLen >= 4) {
                est += 0.12f;
            }
        }
    }

    if (trump != Suit::None && trumpLen > 0 && n >= 4) {
        est += 0.22f * static_cast<float>(std::min(trumpLen, voidSideSuits));
    }

    // 4. Поправка на позицию и давление заказов стола
    const PlayerId firstBidder = static_cast<PlayerId>((st.dealer + 1) % kNumPlayers);
    if (p == firstBidder) {
        est += 0.10f;
    }

    if (st.bidsMadeInDeal > 0) {
        int sumBids = 0;
        for (int q = 0; q < kNumPlayers; ++q) {
            if (st.bids[q] >= 0) sumBids += st.bids[q];
        }
        const float expectedSoFar = static_cast<float>(n) * (static_cast<float>(st.bidsMadeInDeal) / 4.0f);
        const float pressure = static_cast<float>(sumBids) - expectedSoFar;
        est -= 0.12f * std::clamp(pressure, -2.5f, 2.5f);
    }

    est = std::clamp(est, 0.0f, static_cast<float>(n));

    // 5. Выбор заказа по максимуму EV с учётом пресета и пульки
    const float sigma = 0.52f + 0.11f * std::sqrt(static_cast<float>(n));
    const float inv2Sig2 = 1.0f / (2.0f * sigma * sigma);

    std::array<float, 10> probs{};
    float sumP = 0.0f;
    for (int k = 0; k <= n; ++k) {
        const float d = static_cast<float>(k) - est;
        const float w = std::exp(-d * d * inv2Sig2);
        probs[static_cast<size_t>(k)] = w;
        sumP += w;
    }
    const float invSumP = (sumP > 0.0f) ? (1.0f / sumP) : 1.0f;
    for (int k = 0; k <= n; ++k) {
        probs[static_cast<size_t>(k)] *= invSumP;
    }

    float pulkaProtect = 0.0f;
    if (st.pulkaClean[p]) {
        const float bonusVal = st.pulkaAllPasses[p]
            ? 450.0f
            : static_cast<float>(std::max<int>(100, st.pulkaMaxScoreExclLast[p]));
        pulkaProtect = 0.35f * bonusVal;
    }

    ActionId bestBid = static_cast<ActionId>(std::countr_zero(legalMask));
    float bestEV = -1e18f;
    const uint64_t tie = Deck::SplitMix64(rngState);

    for (int b = 0; b <= n; ++b) {
        if (((legalMask >> b) & 1ULL) == 0ULL) continue;

        float ev = 0.0f;
        for (int k = 0; k <= n; ++k) {
            const float sc = static_cast<float>(rules::CalculateDealScore(st.rules, b, k, n));
            ev += probs[static_cast<size_t>(k)] * sc;
        }

        ev += probs[static_cast<size_t>(b)] * pulkaProtect;
        if (st.pulkaClean[p] && st.pulkaAllPasses[p] && b == 0) {
            ev += probs[0] * 120.0f;
        }

        const float jitter = static_cast<float>((tie >> ((b % 8) * 8)) & 0xFFU) * 1e-4f;
        if (ev + jitter > bestEV) {
            bestEV = ev + jitter;
            bestBid = static_cast<ActionId>(b);
        }
    }

    return bestBid;
}

inline ActionId HeuristicTrickPlay(
    const GameState& st,
    const ActionMask legalMask,
    uint64_t& /*rngState*/) noexcept {
    const PlayerId p = st.currentPlayer;
    const HandMask myHand = st.hands[p];
    const HandMask fullDeck = rules::GetFullDeckMask(st.rules);
    const HandMask upcardMask = (st.upcard != kInvalidCard && st.upcard < kMaxDeckSize)
        ? (1ULL << st.upcard)
        : 0ULL;
    const HandMask unseenMask = fullDeck & ~(myHand | st.playedCardsHistory | upcardMask);
    const Suit trump = st.trumpSuit;

    const int bid = st.bids[p];
    const int taken = st.tricksTaken[p];
    const int need = bid - taken;
    const int remTricks = st.cardsInDeal - st.tricksCompletedInDeal;
    const bool isLead = (st.cardsInCurrentTrick == 0U);

    bool wantTrick = false;
    bool urgentTrick = false;
    if (need > 0 && need <= remTricks) {
        wantTrick = true;
        urgentTrick = (need == remTricks);
    } else if (need == 0) {
        wantTrick = false;
    } else {
        if (st.rules == RulePreset::Popular && bid > 0 && taken == 0) {
            wantTrick = true;
            urgentTrick = true;
        } else if (st.rules == RulePreset::Minus && need > remTricks) {
            wantTrick = true;
            urgentTrick = true;
        } else {
            wantTrick = true;
        }
    }

    const PlayerId partner = static_cast<PlayerId>((p + 2) % kNumPlayers);
    const bool isPairs = (st.teamMode == TeamMode::Pairs);
    const int partnerNeed = isPairs ? (st.bids[partner] - st.tricksTaken[partner]) : 0;

    auto isMasterCard = [&](const CardId cid) noexcept -> bool {
        const Card& c = GetCard(cid);
        if (c.IsJoker()) return true;
        return (GetHigherCardsInSuitMask(c.suit, c.rank) & unseenMask) == 0ULL;
    };

    auto anyOppVoidInSuit = [&](const Suit s) noexcept -> bool {
        if (s == Suit::None) return false;
        const HandMask sm = GetSuitMask(s);
        for (PlayerId q = 0; q < kNumPlayers; ++q) {
            if (q == p) continue;
            if ((st.impossibleCards[q] & sm) == sm) return true;
        }
        return false;
    };

    // Ветка A: Первый ход во взятку (Lead)
    if (isLead) {
        const bool hasJoker = ((legalMask & kJokerLeadActionsMask) != 0ULL);

        if (wantTrick) {
            ActionId bestNormal = kInvalidAction;
            float bestNormalScore = -1e9f;

            HandMask cards = legalMask & ~kJokerLeadActionsMask;
            while (cards != 0ULL) {
                const CardId cid = LowestCardInMask(cards);
                cards &= (cards - 1ULL);
                const Card& c = GetCard(cid);
                const bool isT = (trump != Suit::None && c.suit == trump);
                const bool master = isMasterCard(cid);
                const int rankVal = static_cast<int>(std::to_underlying(c.rank));
                const int unseenInSuit = std::popcount(unseenMask & GetSuitMask(c.suit));
                const int unseenTrumps = (trump != Suit::None) ? std::popcount(unseenMask & GetSuitMask(trump)) : 0;

                float sc = static_cast<float>(rankVal) * 1.2f;
                if (master) {
                    if (isT || trump == Suit::None || unseenTrumps == 0) {
                        sc += 45.0f;
                    } else {
                        sc += anyOppVoidInSuit(c.suit) ? 18.0f : 34.0f;
                    }
                    if (unseenInSuit >= 2) sc += 3.0f;
                } else {
                    if (isT) sc -= 8.0f;
                    if (unseenInSuit == 0 && !isT) sc -= 15.0f;
                }

                if (sc > bestNormalScore) {
                    bestNormalScore = sc;
                    bestNormal = cid;
                }
            }

            if (bestNormal != kInvalidAction && bestNormalScore >= 32.0f) {
                return bestNormal;
            }

            if (hasJoker && (urgentTrick || bestNormal == kInvalidAction || bestNormalScore < 20.0f)) {
                if (trump != Suit::None) {
                    const ActionId jTrump = static_cast<ActionId>(kActionJokerLeadHighBase + std::to_underlying(trump));
                    if (((legalMask >> jTrump) & 1ULL) != 0ULL) {
                        return jTrump;
                    }
                }
                ActionId bestJ = kActionJokerLeadHighBase;
                int bestUnseen = -1;
                for (uint8_t s = 0; s < 4U; ++s) {
                    const ActionId a = static_cast<ActionId>(kActionJokerLeadHighBase + s);
                    if (((legalMask >> a) & 1ULL) == 0ULL) continue;
                    const Suit suit = static_cast<Suit>(s);
                    int u = std::popcount(unseenMask & GetSuitMask(suit));
                    if (anyOppVoidInSuit(suit) && trump != Suit::None) u -= 10;
                    if (u > bestUnseen) {
                        bestUnseen = u;
                        bestJ = a;
                    }
                }
                return bestJ;
            }

            if (bestNormal != kInvalidAction) {
                return bestNormal;
            }
            return static_cast<ActionId>(std::countr_zero(legalMask));
        }

        // Режим сброса на первом ходе
        ActionId bestDump = kInvalidAction;
        float bestDumpScore = -1e9f;

        HandMask cards = legalMask & ~kJokerLeadActionsMask;
        while (cards != 0ULL) {
            const CardId cid = LowestCardInMask(cards);
            cards &= (cards - 1ULL);
            const Card& c = GetCard(cid);
            const bool isT = (trump != Suit::None && c.suit == trump);
            const int rankVal = static_cast<int>(std::to_underlying(c.rank));
            const int higherOut = std::popcount(GetHigherCardsInSuitMask(c.suit, c.rank) & unseenMask);

            float sc = -static_cast<float>(rankVal) * 2.5f;
            if (isT) sc -= 14.0f;
            if (higherOut == 0) {
                sc -= 40.0f;
            } else {
                sc += static_cast<float>(std::min(higherOut, 4)) * 6.0f;
            }

            if (sc > bestDumpScore) {
                bestDumpScore = sc;
                bestDump = cid;
            }
        }

        if (hasJoker && (bestDump == kInvalidAction || bestDumpScore < -20.0f || remTricks == 1)) {
            ActionId bestJLow = kActionJokerLeadLowBase;
            int bestScore = -100;
            for (uint8_t s = 0; s < 4U; ++s) {
                const ActionId a = static_cast<ActionId>(kActionJokerLeadLowBase + s);
                if (((legalMask >> a) & 1ULL) == 0ULL) continue;
                int u = std::popcount(unseenMask & GetSuitMask(static_cast<Suit>(s)));
                if (static_cast<Suit>(s) == trump) u += 2;
                if (u > bestScore) {
                    bestScore = u;
                    bestJLow = a;
                }
            }
            return bestJLow;
        }

        return (bestDump != kInvalidAction) ? bestDump : static_cast<ActionId>(std::countr_zero(legalMask));
    }

    // Ветка B: Ответ во взятку (Follow)
    const PlayerId curWinner = rules::DetermineTrickWinnerN(st.currentTrick, st.cardsInCurrentTrick, trump);
    const bool partnerWinning = isPairs && (curWinner == partner);
    const bool isLastToAct = (st.cardsInCurrentTrick == 3U);

    bool effectiveWantTrick = wantTrick;
    if (partnerWinning && partnerNeed > 0 && !urgentTrick) {
        effectiveWantTrick = false;
    }
    if (partnerWinning && partnerNeed == 0 && (need > 0 || (need < 0 && st.rules != RulePreset::Minus))) {
        effectiveWantTrick = true;
    }

    ActionId bestWinNormal = kInvalidAction;
    float bestWinScore = -1e9f;
    ActionId bestLoseNormal = kInvalidAction;
    float bestLoseScore = -1e9f;

    std::array<TrickCard, kNumPlayers> simTrick = st.currentTrick;
    const uint8_t nextCnt = static_cast<uint8_t>(st.cardsInCurrentTrick + 1U);

    HandMask normalCards = legalMask & ~kJokerFollowActionsMask;
    while (normalCards != 0ULL) {
        const CardId cid = LowestCardInMask(normalCards);
        normalCards &= (normalCards - 1ULL);
        const Card& c = GetCard(cid);

        simTrick[st.cardsInCurrentTrick] = TrickCard{
            p, CardPlay{.cardId = cid, .jokerAction = JokerAction::None, .requestedSuit = Suit::None}
        };
        const bool winsNow = (rules::DetermineTrickWinnerN(simTrick, nextCnt, trump) == p);
        const bool isT = (trump != Suit::None && c.suit == trump);
        const bool master = isMasterCard(cid);
        const int rankVal = static_cast<int>(std::to_underlying(c.rank));

        if (winsNow) {
            float sc = 0.0f;
            if (isLastToAct) {
                sc = 100.0f - static_cast<float>(rankVal) * 2.0f - (isT ? 15.0f : 0.0f);
            } else {
                sc = 50.0f + (master ? 25.0f : static_cast<float>(rankVal) * 1.5f) - (isT ? 6.0f : 0.0f);
            }
            if (sc > bestWinScore) {
                bestWinScore = sc;
                bestWinNormal = cid;
            }
        } else {
            float sc = 0.0f;
            if (effectiveWantTrick) {
                sc = 50.0f - static_cast<float>(rankVal) * 2.5f - (isT ? 25.0f : 0.0f) - (master ? 35.0f : 0.0f);
            } else {
                sc = 50.0f + static_cast<float>(rankVal) * 3.0f + (master ? 20.0f : 0.0f) + (isT ? 10.0f : 0.0f);
            }
            if (sc > bestLoseScore) {
                bestLoseScore = sc;
                bestLoseNormal = cid;
            }
        }
    }

    const bool canJokerWin = ((legalMask >> kActionJokerPlayWin) & 1ULL) != 0ULL;
    const bool canJokerLow = ((legalMask >> kActionJokerPlayLow) & 1ULL) != 0ULL;

    if (effectiveWantTrick) {
        if (bestWinNormal != kInvalidAction && (isLastToAct || bestWinScore >= 60.0f || !canJokerWin)) {
            return bestWinNormal;
        }
        if (canJokerWin && (urgentTrick || isLastToAct || bestWinNormal == kInvalidAction)) {
            return kActionJokerPlayWin;
        }
        if (bestWinNormal != kInvalidAction) {
            return bestWinNormal;
        }
        if (bestLoseNormal != kInvalidAction) {
            return bestLoseNormal;
        }
        return canJokerWin ? kActionJokerPlayWin : static_cast<ActionId>(std::countr_zero(legalMask));
    }

    if (bestLoseNormal != kInvalidAction) {
        return bestLoseNormal;
    }
    if (canJokerLow) {
        return kActionJokerPlayLow;
    }
    if (bestWinNormal != kInvalidAction) {
        ActionId lowestWin = bestWinNormal;
        int minRank = 100;
        HandMask m = legalMask & ~kJokerFollowActionsMask;
        while (m != 0ULL) {
            const CardId cid = LowestCardInMask(m);
            m &= (m - 1ULL);
            const int r = static_cast<int>(std::to_underlying(GetCard(cid).rank));
            if (r < minRank) {
                minRank = r;
                lowestWin = cid;
            }
        }
        return lowestWin;
    }

    return static_cast<ActionId>(std::countr_zero(legalMask));
}

} // namespace

ActionId HeuristicBot::SelectAction(const JokerEngine& env, uint64_t& rngState) noexcept {
    const ActionMask mask = env.GetLegalActionsMask();
    if (mask == 0ULL) return 0U;

    if (std::popcount(mask) == 1) {
        return static_cast<ActionId>(std::countr_zero(mask));
    }

    const GameState& st = env.GetState();
    if (!st.IsHandKnown(st.currentPlayer)) {
        return PickRandomLegalAction(mask, rngState);
    }

    if (st.phase == Phase::ChoosingTrump) {
        return HeuristicChooseTrump(st, mask, rngState);
    }
    if (st.phase == Phase::Bidding) {
        return HeuristicBid(st, mask, rngState);
    }
    if (st.phase == Phase::TrickPlay) {
        return HeuristicTrickPlay(st, mask, rngState);
    }
    return PickRandomLegalAction(mask, rngState);
}

} // namespace joker::engine