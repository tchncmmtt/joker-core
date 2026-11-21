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

inline float CalcPulkaProtectionWeight(const GameState& st, const PlayerId p) noexcept {
    if (!st.pulkaClean[p]) {
        return 0.0f;
    }

    const int totalDeals = GetTotalDeals(st.mode);
    const int dealIdx = std::clamp(static_cast<int>(st.dealNumber), 0, totalDeals - 1);
    const auto& sched = GetScheduleEntry(st.mode, dealIdx);
    const float pulkaLen = sched.isNineCardDeal ? 4.0f : 8.0f;
    const float progress = static_cast<float>(sched.roundInPulka + 1) / pulkaLen;
    const float stageMult = sched.isPulkaEnd ? 0.92f : (0.28f + 0.48f * progress);

    const float myBonus = st.pulkaAllPasses[p]
        ? 450.0f
        : static_cast<float>(std::max<int>(100, st.pulkaMaxScoreExclLast[p]));
    const float myPenalty = static_cast<float>(std::max<int>(50, st.pulkaMaxNonPassScoreExclLast[p]));

    if (st.pulkaMode == PulkaMode::Soft) {
        const PlayerId over = static_cast<PlayerId>((p + kNumPlayers - 1) % kNumPlayers);
        const float swing = st.pulkaClean[over] ? myPenalty : myBonus;
        return stageMult * swing;
    }

    if (st.teamMode == TeamMode::Pairs) {
        const PlayerId partner = static_cast<PlayerId>((p + 2) % kNumPlayers);
        const PlayerId e1 = static_cast<PlayerId>((p + 1) % kNumPlayers);
        const PlayerId e2 = static_cast<PlayerId>((p + 3) % kNumPlayers);
        const bool enemyClean = st.pulkaClean[e1] || st.pulkaClean[e2];
        const bool partnerClean = st.pulkaClean[partner];
        float swing = 0.0f;
        if (!enemyClean) {
            swing += myBonus;
            if (!partnerClean) {
                swing += 0.35f * static_cast<float>(
                    st.pulkaMaxNonPassScoreExclLast[e1] + st.pulkaMaxNonPassScoreExclLast[e2]);
            }
        } else if (!partnerClean) {
            swing += myPenalty;
        }
        return stageMult * swing;
    }

    // Solo + Strict
    int otherClean = 0;
    float oppPenaltySum = 0.0f;
    for (PlayerId q = 0; q < kNumPlayers; ++q) {
        if (q == p) continue;
        if (st.pulkaClean[q]) {
            ++otherClean;
        } else {
            oppPenaltySum += static_cast<float>(st.pulkaMaxNonPassScoreExclLast[q]);
        }
    }

    float swing = 0.0f;
    if (otherClean == 0) {
        swing = myBonus + 0.25f * oppPenaltySum;
    } else if (otherClean == 1) {
        swing = myPenalty + 0.35f * myBonus;
    } else {
        swing = myPenalty;
    }
    return stageMult * swing;
}

// ============================================================================
// Фаза 1: Выбор козыря по 3 картам на девятках
// ============================================================================
inline ActionId HeuristicChooseTrump(
    const GameState& st,
    const ActionMask legalMask,
    uint64_t& rngState) noexcept {
    const PlayerId p = st.currentPlayer;
    const HandMask hand = st.hands[p];
    const int nJokers = std::popcount(hand & kJokerMask);
    const bool preferPass = st.pulkaClean[p] && st.pulkaAllPasses[p];

    int numAces = 0;
    int numKings = 0;
    int numQueens = 0;
    int maxSuitLen = 0;
    int highCardPts = 0;

    std::array<float, 5> scores{-1e9f, -1e9f, -1e9f, -1e9f, -1e9f};

    for (uint8_t s = 0; s < 4U; ++s) {
        const HandMask sCards = hand & GetSuitMask(static_cast<Suit>(s));
        const int cnt = std::popcount(sCards);
        if (cnt > maxSuitLen) maxSuitLen = cnt;

        float val = (cnt == 3) ? 4.4f : ((cnt == 2) ? 2.2f : ((cnt == 1) ? 0.4f : -2.2f));
        HandMask m = sCards;
        while (m != 0ULL) {
            const CardId cid = LowestCardInMask(m);
            m &= (m - 1ULL);
            const Rank r = GetCard(cid).rank;
            switch (r) {
                case Rank::Ace:   val += 2.3f; ++numAces;   highCardPts += 4; break;
                case Rank::King:  val += 1.35f; ++numKings; highCardPts += 3; break;
                case Rank::Queen: val += 0.75f; ++numQueens; highCardPts += 2; break;
                case Rank::Jack:  val += 0.35f; highCardPts += 1; break;
                case Rank::Ten:   val += 0.15f; break;
                default: break;
            }
        }
        if (nJokers > 0 && cnt >= 2) {
            val += 0.4f * static_cast<float>(nJokers);
        }
        // Если идём на «все пасы» и рука слабая — избегаем объявлять длинную масть козырем
        if (preferPass && nJokers == 0 && highCardPts <= 2) {
            val = -val * 0.4f + (cnt == 0 ? 1.8f : 0.0f);
        }
        scores[s] = val;
    }

    // Бескозырка (Suit::None = 4)
    float ntScore = 1.75f * static_cast<float>(nJokers)
                  + 1.50f * static_cast<float>(numAces)
                  + 0.60f * static_cast<float>(numKings)
                  + 0.25f * static_cast<float>(numQueens)
                  - 0.45f;
    if (nJokers >= 2) {
        ntScore += 1.6f;
    } else if (nJokers == 1 && maxSuitLen <= 1) {
        ntScore += 1.0f;
    }
    if (preferPass && nJokers == 0 && highCardPts <= 2) {
        ntScore = 2.4f;
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

// ============================================================================
// Фаза 2: Торговля (Заказ взяток)
// ============================================================================
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
    const PlayerId firstBidder = static_cast<PlayerId>((st.dealer + 1) % kNumPlayers);
    const bool isFirst = (p == firstBidder);

    float est = 0.0f;

    // 1. Джокеры
    const int nJokers = std::popcount(hand & kJokerMask);
    const float jokerVal = (trump == Suit::None) ? 0.98f : 0.94f;
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

            if (n == 1) {
                // На 1 карте любой козырь силён, особенно если выше него мало козырей в раздаче
                est += std::clamp(0.95f - 0.14f * effHigher, 0.25f, 0.95f);
            } else if (effHigher < 0.35f) {
                est += 0.97f;
            } else if (effHigher < 1.15f) {
                est += (trumpLen >= 2) ? 0.84f : 0.58f;
            } else if (effHigher < 2.15f) {
                est += (trumpLen >= 3) ? 0.68f : 0.34f;
            } else if (effHigher < 3.25f) {
                est += (trumpLen >= 4) ? 0.48f : 0.16f;
            } else if (trumpLen >= 4) {
                est += 0.28f;
            }
        }
        // Длина козырной масти (контроль стола)
        if (trumpLen >= 3 && n >= 4) {
            est += 0.28f * static_cast<float>(trumpLen - 2);
            if (trumpLen >= 5) {
                est += 0.25f * static_cast<float>(trumpLen - 4);
            }
        }
    }

    // 3. Побочные масти (или все 4 при Б/К)
    int voidSideSuits = 0;
    int singletonSideSuits = 0;
    const float trumpCutRisk = (trump == Suit::None) ? 0.0f : ((n >= 7) ? 0.24f : ((n >= 4) ? 0.15f : 0.08f));

    for (uint8_t s = 0; s < 4U; ++s) {
        const Suit suit = static_cast<Suit>(s);
        if (suit == trump) continue;
        const HandMask sCards = hand & GetSuitMask(suit);
        const int sLen = std::popcount(sCards);
        if (sLen == 0) {
            ++voidSideSuits;
            continue;
        }
        if (sLen == 1) {
            ++singletonSideSuits;
        }

        HandMask m = sCards;
        while (m != 0ULL) {
            const CardId cid = LowestCardInMask(m);
            m &= (m - 1ULL);
            const Rank r = GetCard(cid).rank;
            const int higherOut = std::popcount(GetHigherCardsInSuitMask(suit, r) & fullDeck & ~hand & ~upcardMask);
            const float effHigher = static_cast<float>(higherOut) * dealFrac;

            if (n == 1) {
                // На 1 карте побочная масть имеет шансы только если мы ходим первыми или масть совпадёт
                const float leadMult = isFirst ? 0.85f : 0.35f;
                const float trumpPenalty = (trump == Suit::None) ? 1.0f : 0.65f;
                est += std::clamp(0.88f - 0.16f * effHigher, 0.02f, 0.88f) * leadMult * trumpPenalty;
            } else if (effHigher < 0.35f) {
                // Мастер (обычно Туз)
                const float cut = (sLen >= 4) ? (trumpCutRisk * 1.35f) : trumpCutRisk;
                const float base = (trump == Suit::None) ? 0.94f : (0.88f * (1.0f - cut));
                est += base;
            } else if (effHigher < 1.15f) {
                // Второй по старшинству (Король с фоской)
                float base = (sLen >= 2) ? 0.56f : 0.18f;
                if (trump != Suit::None) base *= (1.0f - trumpCutRisk);
                est += base;
            } else if (effHigher < 2.15f) {
                float base = (sLen >= 3) ? 0.25f : 0.05f;
                if (trump != Suit::None) base *= (1.0f - trumpCutRisk);
                est += base;
            } else if (trump == Suit::None && sLen >= 4) {
                est += 0.16f;
            }
        }
    }

    // Бонус за ренонсы/синглеты при наличии козырей
    if (trump != Suit::None && trumpLen > 0 && n >= 4) {
        const float ruffPower = static_cast<float>(voidSideSuits) * 0.26f
                              + static_cast<float>(singletonSideSuits) * 0.10f;
        est += std::min(static_cast<float>(trumpLen) * 0.45f, ruffPower);
    }

    // 4. Позиция и уже сделанные заказы стола
    if (isFirst && n >= 2) {
        est += 0.12f;
    }

    int sumBidsSoFar = 0;
    for (int q = 0; q < kNumPlayers; ++q) {
        if (st.bids[q] >= 0) sumBidsSoFar += st.bids[q];
    }

    const int remBiddersAfterMe = (kNumPlayers - 1) - st.bidsMadeInDeal;
    if (st.bidsMadeInDeal > 0 && remBiddersAfterMe > 0) {
        const float expectedSoFar = static_cast<float>(n) * (static_cast<float>(st.bidsMadeInDeal) / 4.0f);
        const float pressure = static_cast<float>(sumBidsSoFar) - expectedSoFar;
        est -= 0.14f * std::clamp(pressure, -2.5f, 2.5f);
    }

    est = std::clamp(est, 0.0f, static_cast<float>(n));

    // 5. Выбор заказа по максимуму EV с учётом пресета, пульки и сдвига баланса для дилера
    const float sigma = (n == 1) ? 0.48f : (0.50f + 0.105f * std::sqrt(static_cast<float>(n)));
    const float inv2Sig2 = 1.0f / (2.0f * sigma * sigma);
    const float pulkaProtect = CalcPulkaProtectionWeight(st, p);
    const bool isDealer = (p == st.dealer);

    ActionId bestBid = static_cast<ActionId>(std::countr_zero(legalMask));
    float bestEV = -1e18f;
    const uint64_t tie = Deck::SplitMix64(rngState);

    for (int b = 0; b <= n; ++b) {
        if (((legalMask >> b) & 1ULL) == 0ULL) continue;

        // Если мы дилер, мы точно знаем итоговый баланс стола (недобор/перебор) для данного b
        float effEst = est;
        if (isDealer) {
            const float tableSurplus = static_cast<float>(sumBidsSoFar + b - n);
            effEst = std::clamp(est - 0.16f * std::clamp(tableSurplus, -3.0f, 3.0f), 0.0f, static_cast<float>(n));
        }

        std::array<float, 10> probs{};
        float sumP = 0.0f;
        for (int k = 0; k <= n; ++k) {
            const float d = static_cast<float>(k) - effEst;
            const float w = std::exp(-d * d * inv2Sig2);
            probs[static_cast<size_t>(k)] = w;
            sumP += w;
        }
        const float invSumP = (sumP > 0.0f) ? (1.0f / sumP) : 1.0f;

        float ev = 0.0f;
        for (int k = 0; k <= n; ++k) {
            const float pk = probs[static_cast<size_t>(k)] * invSumP;
            const float sc = static_cast<float>(rules::CalculateDealScore(st.rules, b, k, n));
            ev += pk * sc;
        }

        const float pExact = probs[static_cast<size_t>(b)] * invSumP;
        ev += pExact * pulkaProtect;
        if (st.pulkaClean[p] && st.pulkaAllPasses[p] && b == 0) {
            ev += pExact * 140.0f;
        }

        const float jitter = static_cast<float>((tie >> ((b % 8) * 8)) & 0xFFU) * 1e-4f;
        if (ev + jitter > bestEV) {
            bestEV = ev + jitter;
            bestBid = static_cast<ActionId>(b);
        }
    }

    return bestBid;
}

// ============================================================================
// Фаза 3: Розыгрыш взятки
// ============================================================================
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
    const bool isLastToAct = (st.cardsInCurrentTrick == 3U);

    const bool isPairs = (st.teamMode == TeamMode::Pairs);
    const PlayerId partner = static_cast<PlayerId>((p + 2) % kNumPlayers);
    const int partnerNeed = isPairs ? (st.bids[partner] - st.tricksTaken[partner]) : 0;

    auto isEnemy = [&](const PlayerId q) noexcept -> bool {
        if (q == p) return false;
        return !isPairs || (q != partner);
    };

    // Оставшиеся игроки, которые ещё будут ходить в этой взятке после нас
    uint8_t remPlayersCount = static_cast<uint8_t>(3U - st.cardsInCurrentTrick);
    std::array<PlayerId, 3> remPlayers{};
    for (uint8_t i = 0; i < remPlayersCount; ++i) {
        remPlayers[i] = static_cast<PlayerId>((p + 1U + i) % kNumPlayers);
    }

    // Проверка: может ли какой-то оставшийся ВРАГ побить нашу карту (мастью старше или козырем при ренонсе)
    auto canRemainingEnemyBeat = [&](const Card& c) noexcept -> bool {
        if (c.IsJoker()) return false;
        const HandMask higherSameSuit = GetHigherCardsInSuitMask(c.suit, c.rank) & unseenMask;
        const HandMask trumpMask = (trump != Suit::None) ? (GetSuitMask(trump) & unseenMask) : 0ULL;
        const HandMask suitMask = GetSuitMask(c.suit);

        for (uint8_t i = 0; i < remPlayersCount; ++i) {
            const PlayerId q = remPlayers[i];
            if (!isEnemy(q)) continue;
            const HandMask qPossible = unseenMask & ~st.impossibleCards[q];
            if ((qPossible & higherSameSuit) != 0ULL) {
                return true;
            }
            if (trump != Suit::None && c.suit != trump) {
                const bool qVoidInSuit = (qPossible & suitMask) == 0ULL;
                const bool qHasTrump = (qPossible & trumpMask) != 0ULL;
                if (qVoidInSuit && qHasTrump) {
                    return true;
                }
            }
        }
        return false;
    };

    auto anyRemainingEnemyVoidInSuit = [&](const Suit s) noexcept -> bool {
        if (s == Suit::None) return false;
        const HandMask sm = GetSuitMask(s);
        for (uint8_t i = 0; i < remPlayersCount; ++i) {
            const PlayerId q = remPlayers[i];
            if (!isEnemy(q)) continue;
            if ((st.impossibleCards[q] & sm) == sm) return true;
        }
        return false;
    };

    auto isMasterCard = [&](const CardId cid) noexcept -> bool {
        const Card& c = GetCard(cid);
        if (c.IsJoker()) return true;
        return (GetHigherCardsInSuitMask(c.suit, c.rank) & unseenMask) == 0ULL;
    };

    // Определяем базовую цель на взятку
    bool wantTrick = false;
    bool urgentTrick = false;
    const bool contractBroken = (need < 0) || (need > remTricks);

    if (need > 0 && need <= remTricks) {
        wantTrick = true;
        urgentTrick = (need == remTricks);
    } else if (need == 0) {
        wantTrick = false;
    } else {
        // Контракт уже сломан
        if (st.rules == RulePreset::Popular && bid > 0 && taken == 0) {
            // Спасаемся от штанги (-200)!
            wantTrick = true;
            urgentTrick = true;
        } else if (st.rules == RulePreset::Minus && need > remTricks) {
            // В Minus каждая недобратая взятка стоит -100
            wantTrick = true;
            urgentTrick = true;
        } else {
            // Перебор: каждая взятка даёт +10 (в Pairs также ориентируемся на партнёра)
            wantTrick = true;
        }
    }

    const int unseenTrumps = (trump != Suit::None) ? std::popcount(unseenMask & GetSuitMask(trump)) : 0;
    const int unseenJokers = std::popcount(unseenMask & kJokerMask);

    // ========================================================================
    // Ветка A: Первый ход во взятку (Lead)
    // ========================================================================
    if (isLead) {
        const bool hasJoker = ((legalMask & kJokerLeadActionsMask) != 0ULL);

        // В Pairs: если наш контракт сломан (или выполнен), а партнёру нужны взятки — помогаем
        if (isPairs && !urgentTrick && partnerNeed > 0 && need <= 0) {
            wantTrick = false;
        }

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
                const bool safeWin = !canRemainingEnemyBeat(c);
                const int rankVal = static_cast<int>(std::to_underlying(c.rank));
                const int unseenInSuit = std::popcount(unseenMask & GetSuitMask(c.suit));
                const int mySuitLen = std::popcount(myHand & GetSuitMask(c.suit));

                float sc = static_cast<float>(rankVal) * 1.1f;

                if (safeWin) {
                    // Гарантированная взятка с учётом impossibleCards и потолков
                    if (isT || trump == Suit::None || unseenTrumps == 0) {
                        sc += 52.0f;
                    } else if (!anyRemainingEnemyVoidInSuit(c.suit)) {
                        sc += 42.0f;
                    } else {
                        sc += 22.0f;
                    }
                    // Если у нас есть несколько мастеров, сперва забираем боковые, пока их не перебили козырем
                    if (!isT && unseenTrumps > 0 && unseenInSuit >= 2) {
                        sc += 5.0f;
                    }
                } else if (master) {
                    sc += anyRemainingEnemyVoidInSuit(c.suit) ? 16.0f : 34.0f;
                } else {
                    // Нет гарантированной победы — пытаемся выбить старшие карты (промоушен)
                    if (isT) sc -= 9.0f;
                    if (unseenInSuit == 0 && !isT) sc -= 18.0f;
                    // Выгоднее ходить из длинной масти с высокой картой (например K+Q), чтобы выбить Туза
                    const int higherOut = std::popcount(GetHigherCardsInSuitMask(c.suit, c.rank) & unseenMask);
                    if (higherOut == 1 && mySuitLen >= 2 && !urgentTrick) {
                        sc += 12.0f;
                    }
                }

                if (sc > bestNormalScore) {
                    bestNormalScore = sc;
                    bestNormal = cid;
                }
            }

            if (bestNormal != kInvalidAction && bestNormalScore >= 36.0f) {
                return bestNormal;
            }

            if (hasJoker && (urgentTrick || bestNormal == kInvalidAction || bestNormalScore < 22.0f)) {
                // 1. Если в игре ещё есть чужие козыри — LeadHighSuit(trump) гарантированно берёт и выбивает старшие козыри
                if (trump != Suit::None && unseenTrumps > 0) {
                    const ActionId jTrump = static_cast<ActionId>(kActionJokerLeadHighBase + std::to_underlying(trump));
                    if (((legalMask >> jTrump) & 1ULL) != 0ULL) {
                        return jTrump;
                    }
                }
                // 2. Иначе выбираем масть без ренонса у врагов, где мы можем выбить старшую карту и продвинуть свою
                ActionId bestJ = kActionJokerLeadHighBase;
                float bestJScore = -1e9f;
                for (uint8_t s = 0; s < 4U; ++s) {
                    const ActionId a = static_cast<ActionId>(kActionJokerLeadHighBase + s);
                    if (((legalMask >> a) & 1ULL) == 0ULL) continue;
                    const Suit suit = static_cast<Suit>(s);
                    const HandMask sMask = GetSuitMask(suit);
                    const int u = std::popcount(unseenMask & sMask);
                    float js = static_cast<float>(u) * 2.0f;

                    if (trump != Suit::None && suit != trump && unseenTrumps > 0) {
                        if (anyRemainingEnemyVoidInSuit(suit)) {
                            js -= 35.0f; // Риск, что наш Джокер перебьют козырем!
                        } else if (u <= 2) {
                            js -= 12.0f;
                        }
                    }
                    // Промоушен: если у нас в этой масти есть карта, под которой ровно 1 чужая старше — выбьем её!
                    const HandMask myS = myHand & sMask;
                    if (myS != 0ULL) {
                        const CardId topMy = HighestCardInMask(myS);
                        const int higherOut = std::popcount(GetHigherCardsInSuitMask(suit, GetCard(topMy).rank) & unseenMask);
                        if (higherOut == 1) js += 14.0f;
                    }
                    if (js > bestJScore) {
                        bestJScore = js;
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

        // Режим сброса на первом ходе (!wantTrick)
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
            const bool safeLose = canRemainingEnemyBeat(c);

            float sc = -static_cast<float>(rankVal) * 2.4f;
            if (isT) sc -= 14.0f;
            if (!safeLose) {
                sc -= 50.0f; // Опасно: можем случайно забрать взятку!
            } else {
                sc += static_cast<float>(std::min(higherOut, 4)) * 6.5f;
                // Если среди врагов есть чистый соперник с need == 0, и он не ренонсный в этой масти — хорошо подкинуть ему
                for (uint8_t i = 0; i < remPlayersCount; ++i) {
                    const PlayerId q = remPlayers[i];
                    if (isEnemy(q) && st.pulkaClean[q] && st.bids[q] == st.tricksTaken[q]) {
                        const HandMask qHigher = (unseenMask & ~st.impossibleCards[q]) & GetHigherCardsInSuitMask(c.suit, c.rank);
                        if (qHigher != 0ULL) sc += 5.0f;
                    }
                }
            }

            if (sc > bestDumpScore) {
                bestDumpScore = sc;
                bestDump = cid;
            }
        }

        if (hasJoker && (bestDump == kInvalidAction || bestDumpScore < -22.0f || remTricks == 1)) {
            ActionId bestJLow = kActionJokerLeadLowBase;
            float bestJLowScore = -1e9f;
            for (uint8_t s = 0; s < 4U; ++s) {
                const ActionId a = static_cast<ActionId>(kActionJokerLeadLowBase + s);
                if (((legalMask >> a) & 1ULL) == 0ULL) continue;
                const Suit suit = static_cast<Suit>(s);
                const int u = std::popcount(unseenMask & GetSuitMask(suit));
                // Критично: LeadLow выигрывает, если НИ У КОГО нет ни заказанной масти, ни козыря!
                const bool anyoneCanTake = (u > 0) || (trump != Suit::None && unseenTrumps > 0) || (unseenJokers > 0);
                float sc = static_cast<float>(u) * 3.0f + (suit == trump ? 4.0f : 0.0f);
                if (!anyoneCanTake) {
                    sc -= 100.0f;
                }
                if (sc > bestJLowScore) {
                    bestJLowScore = sc;
                    bestJLow = a;
                }
            }
            return bestJLow;
        }

        return (bestDump != kInvalidAction) ? bestDump : static_cast<ActionId>(std::countr_zero(legalMask));
    }

    // ========================================================================
    // Ветка B: Ответ во взятку (Follow)
    // ========================================================================
    const PlayerId curWinner = rules::DetermineTrickWinnerN(st.currentTrick, st.cardsInCurrentTrick, trump);
    const bool partnerWinning = isPairs && (curWinner == partner);
    const bool curWinnerIsCleanOverbidTarget =
        isEnemy(curWinner) && st.pulkaClean[curWinner] && (st.bids[curWinner] == st.tricksTaken[curWinner]);

    bool effectiveWantTrick = wantTrick;

    // Командная координация (Pairs)
    if (isPairs) {
        if (partnerWinning && partnerNeed > 0 && !urgentTrick) {
            effectiveWantTrick = false;
        } else if (partnerWinning && partnerNeed == 0 && (need > 0 || contractBroken)) {
            // Забираем взятку у партнёра, чтобы не сломать ему контракт!
            effectiveWantTrick = true;
        }
    }

    // Подсад соперника (Spoiler): если наш контракт уже сломан (перебор) и не горит штанга/Minus,
    // а взятку сейчас держит чистый соперник с need == 0 — ОТДАЁМ ему взятку, чтобы сломать ему пульку!
    if (contractBroken && !urgentTrick && curWinnerIsCleanOverbidTarget) {
        effectiveWantTrick = false;
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
        const bool safeWin = winsNow && (isLastToAct || !canRemainingEnemyBeat(c));
        const int rankVal = static_cast<int>(std::to_underlying(c.rank));
        const int mySuitLen = std::popcount(myHand & GetSuitMask(c.suit));

        if (winsNow) {
            float sc = 0.0f;
            if (isLastToAct) {
                // Последний ход: забираем самой дешёвой победной картой, бережём козыри и мастера
                sc = 110.0f - static_cast<float>(rankVal) * 2.2f - (isT ? 18.0f : 0.0f) - (master ? 10.0f : 0.0f);
            } else if (safeWin) {
                // Никто из оставшихся врагов не перебьёт — берём минимальной безопасной картой
                sc = 85.0f - static_cast<float>(rankVal) * 1.2f - (isT ? 10.0f : 0.0f);
            } else {
                // Нас могут перебить сзади: предпочитаем бить повыше, чтобы выбить старшие или удержать
                sc = 45.0f + (master ? 28.0f : static_cast<float>(rankVal) * 1.8f) - (isT ? 6.0f : 0.0f);
            }
            if (sc > bestWinScore) {
                bestWinScore = sc;
                bestWinNormal = cid;
            }
        } else {
            float sc = 0.0f;
            if (effectiveWantTrick) {
                // Не можем забрать, но взятки ещё нужны -> бережём козыри и мастеров, создаём ренонс в слабой короткой масти
                sc = 50.0f - static_cast<float>(rankVal) * 2.5f - (isT ? 28.0f : 0.0f) - (master ? 38.0f : 0.0f);
                if (!isT && mySuitLen == 1 && !master && trump != Suit::None && (myHand & GetSuitMask(trump)) != 0ULL) {
                    sc += 10.0f; // Создаём ренонс для будущего перебития козырем
                }
            } else {
                // Не хотим брать взятку -> скидываем самые опасные высокие карты, мастеров и козыри под чужую взятку!
                sc = 50.0f + static_cast<float>(rankVal) * 3.2f + (master ? 26.0f : 0.0f) + (isT ? 14.0f : 0.0f);
                if (mySuitLen == 1 && rankVal >= 5) {
                    sc += 8.0f; // Избавляемся от опасного синглета-фигуры
                }
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
        if (bestWinNormal != kInvalidAction && (isLastToAct || bestWinScore >= 72.0f || !canJokerWin)) {
            return bestWinNormal;
        }
        if (canJokerWin && (urgentTrick || isLastToAct || bestWinNormal == kInvalidAction || bestWinScore < 52.0f)) {
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

    // Не хотим брать взятку (!effectiveWantTrick)
    if (bestLoseNormal != kInvalidAction) {
        return bestLoseNormal;
    }
    if (canJokerLow) {
        return kActionJokerPlayLow;
    }

    // Вынуждены побить (все обычные карты выигрывают, и нет PlayLow Джокера)
    if (bestWinNormal != kInvalidAction) {
        ActionId chosen = bestWinNormal;
        int bestR = isLastToAct ? -1 : 100;
        HandMask m = legalMask & ~kJokerFollowActionsMask;
        while (m != 0ULL) {
            const CardId cid = LowestCardInMask(m);
            m &= (m - 1ULL);
            const int r = static_cast<int>(std::to_underlying(GetCard(cid).rank));
            // Если мы последние и точно забираем — сбрасываем САМУЮ СТАРШУЮ опасную карту;
            // если за нами ещё ходят — ставим САМУЮ МЛАДШУЮ в надежде, что нас перебьют!
            if (isLastToAct ? (r > bestR) : (r < bestR)) {
                bestR = r;
                chosen = cid;
            }
        }
        return chosen;
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