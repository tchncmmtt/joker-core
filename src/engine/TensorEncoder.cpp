#include "engine/TensorEncoder.hpp"
#include "core/Rules.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <utility>

namespace joker::engine {

void WriteOracleTensor(
    const core::GameState& state,
    const core::PlayerId observer,
    float* outOracle) noexcept {
    std::fill_n(outOracle, kOracleTensorSize, 0.0f);
    for (int rel = 0; rel < core::kNumPlayers; ++rel) {
        const core::PlayerId absP = static_cast<core::PlayerId>(
            (static_cast<int>(observer) + rel) % core::kNumPlayers);
        const core::HandMask h = state.hands[absP];
        float* dst = outOracle + static_cast<size_t>(rel) * core::kMaxDeckSize;
        for (size_t id = 0; id < core::kMaxDeckSize; ++id) {
            dst[id] = ((h & (1ULL << id)) != 0ULL) ? 1.0f : 0.0f;
        }
    }
}

} // namespace joker::engine

namespace joker::core {

namespace {

constexpr int RelSeat(const PlayerId absSeat, const PlayerId observer) noexcept {
    return (static_cast<int>(absSeat) - static_cast<int>(observer) + kNumPlayers) % kNumPlayers;
}

constexpr PlayerId AbsSeat(const int relSeat, const PlayerId observer) noexcept {
    return static_cast<PlayerId>((static_cast<int>(observer) + relSeat) % kNumPlayers);
}

constexpr float CalcRawBonusValue(const GameStateView& v, const PlayerId absSeat) noexcept {
    return v.pulkaAllPasses[absSeat]
        ? 450.0f
        : static_cast<float>(v.pulkaMaxScoreExclLast[absSeat]);
}

constexpr float CalcRawPenaltyValue(const GameStateView& v, const PlayerId absSeat) noexcept {
    return static_cast<float>(v.pulkaMaxNonPassScoreExclLast[absSeat]);
}

constexpr std::array<float, kNumPlayers> ComputePulkaDeltas(
    const GameStateView& v,
    const std::array<bool, kNumPlayers>& clean) noexcept {
    std::array<float, kNumPlayers> deltas{0.0f, 0.0f, 0.0f, 0.0f};

    if (v.pulkaMode == PulkaMode::Soft) {
        for (int p = 0; p < kNumPlayers; ++p) {
            const int over = (p + kNumPlayers - 1) % kNumPlayers;
            if (clean[p]) {
                if (!clean[over]) {
                    deltas[p] = CalcRawBonusValue(v, static_cast<PlayerId>(p));
                }
            } else if (clean[over]) {
                deltas[p] = -CalcRawPenaltyValue(v, static_cast<PlayerId>(p));
            }
        }
        return deltas;
    }

    if (v.teamMode == TeamMode::Pairs) {
        const bool teamAClean = clean[0] || clean[2];
        const bool teamBClean = clean[1] || clean[3];
        if (!teamAClean && !teamBClean) {
            return deltas;
        }
        const bool bothClean = teamAClean && teamBClean;
        for (int p = 0; p < kNumPlayers; ++p) {
            const bool ownClean = (p == 0 || p == 2) ? teamAClean : teamBClean;
            if (clean[p]) {
                if (!bothClean) {
                    deltas[p] = CalcRawBonusValue(v, static_cast<PlayerId>(p));
                }
            } else if (!ownClean) {
                deltas[p] = -CalcRawPenaltyValue(v, static_cast<PlayerId>(p));
            }
        }
        return deltas;
    }

    int numClean = 0;
    for (int p = 0; p < kNumPlayers; ++p) {
        if (clean[p]) ++numClean;
    }
    if (numClean == 0) {
        return deltas;
    }
    for (int p = 0; p < kNumPlayers; ++p) {
        if (clean[p]) {
            if (numClean == 1) {
                deltas[p] = CalcRawBonusValue(v, static_cast<PlayerId>(p));
            }
        } else {
            deltas[p] = -CalcRawPenaltyValue(v, static_cast<PlayerId>(p));
        }
    }
    return deltas;
}

constexpr float RelativeBalanceFromDeltas(
    const std::array<float, kNumPlayers>& deltas,
    const PlayerId me,
    const TeamMode teamMode) noexcept {
    if (teamMode == TeamMode::Pairs) {
        const PlayerId partner = AbsSeat(2, me);
        const PlayerId enemy1 = AbsSeat(1, me);
        const PlayerId enemy2 = AbsSeat(3, me);
        return (deltas[me] + deltas[partner]) - (deltas[enemy1] + deltas[enemy2]);
    }
    const float sumAll = deltas[0] + deltas[1] + deltas[2] + deltas[3];
    const float oppMean = (sumAll - deltas[me]) / 3.0f;
    return deltas[me] - oppMean;
}

struct HorizonInfo {
    int remainingTricksInDeal{0};
    int remainingDealsInPulka{0};
    int remainingCardsInPulka{0};
    int nextDealCards{0};
    int maxFutureNonLastDealCards{0};
    int remainingPulkasAfterCurrent{0};
    int remainingMatchCards{0};
    bool isPulkaEnd{false};
    bool isNinesPulka{false};
};

constexpr HorizonInfo ComputeHorizon(const GameStateView& v) noexcept {
    HorizonInfo h{};
    if (v.phase == Phase::MatchFinished) {
        return h;
    }

    const int totalDeals = GetTotalDeals(v.mode);
    const int totalPulkas = GetPulkasCount(v.mode);
    const int dealIdx = std::clamp(v.dealNumber, 0, totalDeals - 1);
    const auto& cur = GetScheduleEntry(v.mode, dealIdx);

    const int completedTricks =
        v.tricksTaken[0] + v.tricksTaken[1] + v.tricksTaken[2] + v.tricksTaken[3];
    h.remainingTricksInDeal = std::max(0, v.cardsInDeal - completedTricks);
    h.isPulkaEnd = cur.isPulkaEnd;
    h.isNinesPulka = cur.isNineCardDeal;
    h.remainingPulkasAfterCurrent = std::max(0, totalPulkas - 1 - cur.pulkaIndex);

    h.remainingCardsInPulka = h.remainingTricksInDeal;
    h.remainingMatchCards = h.remainingTricksInDeal;

    if (!cur.isPulkaEnd && (dealIdx + 1) < totalDeals) {
        h.nextDealCards = GetScheduleEntry(v.mode, dealIdx + 1).cardsPerPlayer;
    }

    for (int d = dealIdx + 1; d < totalDeals; ++d) {
        const auto& e = GetScheduleEntry(v.mode, d);
        h.remainingMatchCards += e.cardsPerPlayer;
        if (e.pulkaIndex == cur.pulkaIndex) {
            h.remainingDealsInPulka++;
            h.remainingCardsInPulka += e.cardsPerPlayer;
            if (!e.isPulkaEnd && e.cardsPerPlayer > h.maxFutureNonLastDealCards) {
                h.maxFutureNonLastDealCards = e.cardsPerPlayer;
            }
        }
    }

    return h;
}

inline void WriteCardMask(float* dst, const HandMask mask) noexcept {
    for (size_t id = 0; id < kMaxDeckSize; ++id) {
        dst[id] = ((mask & (1ULL << id)) != 0ULL) ? 1.0f : 0.0f;
    }
}

inline void WriteBlockA(const GameStateView& v, const HorizonInfo& h, float* out) noexcept {
    if (v.phase != Phase::MatchFinished) {
        out[std::to_underlying(v.phase)] = 1.0f;
    }
    out[3] = static_cast<float>(v.cardsInDeal) / 9.0f;
    out[4] = static_cast<float>(h.remainingTricksInDeal) / 9.0f;

    const int relDealer = RelSeat(v.dealerPlayer, v.observerPlayer);
    out[5 + relDealer] = 1.0f;

    const int firstBidder = (static_cast<int>(v.dealerPlayer) + 1) % kNumPlayers;
    const int biddingOrder = (static_cast<int>(v.observerPlayer) - firstBidder + kNumPlayers) % kNumPlayers;
    out[9] = static_cast<float>(biddingOrder) / 3.0f;

    out[10] = static_cast<float>(h.remainingDealsInPulka) / 8.0f;
    out[11] = static_cast<float>(h.remainingCardsInPulka) / 36.0f;
    out[12] = h.isPulkaEnd ? 1.0f : 0.0f;
    out[13] = h.isNinesPulka ? 1.0f : 0.0f;
    out[14] = static_cast<float>(h.nextDealCards) / 9.0f;
    out[15] = static_cast<float>(h.maxFutureNonLastDealCards) / 9.0f;
    out[16] = static_cast<float>(h.remainingPulkasAfterCurrent) / 3.0f;
    out[17] = static_cast<float>(h.remainingMatchCards) / 144.0f;
}

inline void WriteBlockBPulkaCommon(
    const GameStateView& v,
    const std::array<float, kNumPlayers>& baseDeltas,
    float* out) noexcept {
    const PlayerId me = v.observerPlayer;
    const float baseBal = RelativeBalanceFromDeltas(baseDeltas, me, v.teamMode);

    for (int rel = 0; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, me);
        out[26 + rel] = v.pulkaClean[absP] ? 1.0f : 0.0f;
        out[30 + rel] = v.pulkaAllPasses[absP] ? 1.0f : 0.0f;
        out[34 + rel] = baseDeltas[absP] / 500.0f;

        float netSwing = 0.0f;
        if (v.pulkaClean[absP]) {
            auto altClean = v.pulkaClean;
            altClean[absP] = false;
            const auto altDeltas = ComputePulkaDeltas(v, altClean);
            netSwing = RelativeBalanceFromDeltas(altDeltas, me, v.teamMode) - baseBal;
        }
        out[38 + rel] = netSwing / 500.0f;
    }
}

inline void WriteBlockC(const GameStateView& v, const HorizonInfo& h, float* out) noexcept {
    const PlayerId me = v.observerPlayer;
    const int remTricks = std::max(1, h.remainingTricksInDeal);
    int sumBids = 0;

    for (int rel = 0; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, me);
        const int b = v.bids[absP];
        const int t = v.tricksTaken[absP];
        const bool hasB = (b >= 0);
        if (hasB) sumBids += b;

        const int need = hasB ? (b - t) : 0;
        const float deficitRatio = hasB
            ? std::clamp(static_cast<float>(need) / static_cast<float>(remTricks), 0.0f, 1.5f)
            : 0.0f;
        const bool broken = hasB && (t > b || need > h.remainingTricksInDeal);
        const bool shangaRisk = hasB && (b > 0) && (t == 0);

        out[42 + rel] = hasB ? (static_cast<float>(b) / 9.0f) : 0.0f;
        out[46 + rel] = hasB ? 1.0f : 0.0f;
        out[50 + rel] = static_cast<float>(t) / 9.0f;
        out[54 + rel] = deficitRatio;
        out[58 + rel] = broken ? 1.0f : 0.0f;
        out[62 + rel] = shangaRisk ? 1.0f : 0.0f;
    }

    out[66] = static_cast<float>(v.cardsInDeal - sumBids) / 9.0f;
    out[67] = static_cast<float>(sumBids) / static_cast<float>(std::max(1, v.cardsInDeal));
    if (v.teamMode == TeamMode::Pairs) {
        const PlayerId partner = AbsSeat(2, me);
        const int myBid = std::max<int>(0, v.bids[me]);
        const int pBid = std::max<int>(0, v.bids[partner]);
        const int myNeed = (v.bids[me] >= 0) ? std::max(0, v.bids[me] - v.tricksTaken[me]) : 0;
        const int pNeed = (v.bids[partner] >= 0) ? std::max(0, v.bids[partner] - v.tricksTaken[partner]) : 0;
        out[68] = static_cast<float>(myBid + pBid) / 18.0f;
        out[69] = static_cast<float>(myNeed + pNeed) / 18.0f;
    }

    if (v.phase == Phase::Bidding && me == v.dealerPlayer) {
        int sumOther = 0;
        for (int p = 0; p < kNumPlayers; ++p) {
            if (p != me && v.bids[p] >= 0) sumOther += v.bids[p];
        }
        const int forbidden = v.cardsInDeal - sumOther;
        if (forbidden >= 0 && forbidden <= v.cardsInDeal && forbidden < 10) {
            out[70 + forbidden] = 1.0f;
        }
    }

    out[80 + std::to_underlying(v.trumpSuit)] = 1.0f;

    Suit reqSuit = Suit::None;
    bool forceHigh = false;
    if (v.phase == Phase::TrickPlay && v.cardsInCurrentTrick > 0U) {
        const auto& leadPlay = v.currentTrick[0].play;
        const Card& leadCard = GetCard(leadPlay.cardId);
        reqSuit = leadCard.IsJoker() ? leadPlay.requestedSuit : leadCard.suit;
        forceHigh = leadCard.IsJoker() && (leadPlay.jokerAction == JokerAction::LeadHighSuit);
    }
    out[85 + std::to_underlying(reqSuit)] = 1.0f;

    out[90] = static_cast<float>(v.cardsInCurrentTrick) / 3.0f;
    out[91] = forceHigh ? 1.0f : 0.0f;
    out[92] = (v.phase == Phase::TrickPlay && v.cardsInCurrentTrick == 0U) ? 1.0f : 0.0f;

    if (v.phase == Phase::TrickPlay && v.cardsInCurrentTrick > 0U) {
        const PlayerId curWinner = rules::DetermineTrickWinnerN(v.currentTrick, v.cardsInCurrentTrick, v.trumpSuit);
        out[93 + RelSeat(curWinner, me)] = 1.0f;
        if (v.teamMode == TeamMode::Pairs) {
            out[97] = (curWinner == AbsSeat(2, me)) ? 1.0f : 0.0f;
        }

        for (uint8_t i = 0; i < v.cardsInCurrentTrick; ++i) {
            if (v.currentTrick[i].player == curWinner) {
                const auto& wp = v.currentTrick[i].play;
                const Card& wc = GetCard(wp.cardId);
                if (wc.IsJoker()) {
                    out[98] = 1.0f;
                    out[100] = 1.0f;
                } else {
                    out[99] = (v.trumpSuit != Suit::None && wc.suit == v.trumpSuit) ? 1.0f : 0.0f;
                    out[100] = static_cast<float>(std::to_underlying(wc.rank)) / 8.0f;
                }
                break;
            }
        }

        out[101 + RelSeat(v.currentTrick[0].player, me)] = 1.0f;

        for (uint8_t i = 0; i < v.cardsInCurrentTrick && i < 3U; ++i) {
            const JokerAction ja = v.currentTrick[i].play.jokerAction;
            if (ja != JokerAction::None) {
                out[105 + static_cast<int>(i) * 4 + (std::to_underlying(ja) - 1)] = 1.0f;
            }
        }
    }
}

inline void WriteBlockD(const GameStateView& v, float* out) noexcept {
    const PlayerId me = v.observerPlayer;
    const HandMask upcardMask = (v.upcard != kInvalidCard && v.upcard < kMaxDeckSize)
        ? (1ULL << v.upcard) : 0ULL;

    HandMask currentTrickMask = 0ULL;
    for (uint8_t i = 0; i < v.cardsInCurrentTrick && i < 3U; ++i) {
        const CardId cid = v.currentTrick[i].play.cardId;
        if (cid < kMaxDeckSize) {
            currentTrickMask |= (1ULL << cid);
        }
    }

    WriteCardMask(out + 120, v.myHand);
    WriteCardMask(out + 158, v.playedCardsHistory);
    WriteCardMask(out + 196, upcardMask);
    WriteCardMask(out + 234, currentTrickMask);

    const HandMask visibleMask = v.myHand | v.playedCardsHistory | upcardMask | currentTrickMask;
    const HandMask fullDeck = rules::GetFullDeckMask(v.rules);
    const HandMask unseenMask = fullDeck & ~visibleMask;

    for (int rel = 1; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, me);
        const uint8_t voids = v.voidSuits[absP];
        const HandMask oppPossible = unseenMask & ~v.impossibleCards[absP];
        const int voidBase = 272 + (rel - 1) * 4;
        const int ceilBase = 284 + (rel - 1) * 4;

        for (int s = 0; s < 4; ++s) {
            out[voidBase + s] = ((voids & (1U << s)) != 0U) ? 1.0f : 0.0f;
            const HandMask suitPossible = oppPossible & GetSuitMask(static_cast<Suit>(s));
            if (suitPossible != 0ULL) {
                const CardId maxCid = HighestCardInMask(suitPossible);
                out[ceilBase + s] = static_cast<float>(std::to_underlying(GetCard(maxCid).rank) + 1) / 9.0f;
            }
        }
    }

    for (int rel = 0; rel < kNumPlayers; ++rel) {
        out[296 + rel] = static_cast<float>(v.jokersPlayedBy[AbsSeat(rel, me)]) / 2.0f;
    }
    out[300] = static_cast<float>(std::popcount(unseenMask & kJokerMask)) / 2.0f;
    out[301] = static_cast<float>(std::popcount(v.myHand & kJokerMask)) / 2.0f;

    const float suitCap = (v.rules == RulePreset::Classic) ? 9.0f : 8.0f;
    for (int s = 0; s < 4; ++s) {
        const Suit suit = static_cast<Suit>(s);
        const float maxSuitCards = (s >= 2) ? 9.0f : suitCap;
        const HandMask sMask = GetSuitMask(suit);
        const HandMask unseenSuit = unseenMask & sMask;
        const HandMask mySuit = v.myHand & sMask;

        out[302 + s] = static_cast<float>(std::popcount(unseenSuit)) / maxSuitCards;
        out[306 + s] = static_cast<float>(std::popcount(mySuit)) / maxSuitCards;

        int masterCount = 0;
        HandMask m = mySuit;
        while (m != 0ULL) {
            const CardId cid = LowestCardInMask(m);
            m &= (m - 1ULL);
            if ((GetHigherCardsInSuitMask(suit, GetCard(cid).rank) & unseenSuit) == 0ULL) {
                ++masterCount;
            }
        }
        out[310 + s] = std::min(1.0f, static_cast<float>(masterCount) / 4.0f);

        if (unseenSuit != 0ULL) {
            const CardId topOpp = HighestCardInMask(unseenSuit);
            out[314 + s] = static_cast<float>(std::to_underlying(GetCard(topOpp).rank) + 1) / 9.0f;
        }
    }

    if (v.phase == Phase::TrickPlay && v.cardsInCurrentTrick > 0U && v.myHand != 0ULL) {
        const int handSize = std::popcount(v.myHand);
        const ActionMask legal = rules::GetLegalActionsMask(v);
        int winCards = 0;
        int loseCards = 0;

        std::array<TrickCard, kNumPlayers> simTrick = v.currentTrick;
        const uint8_t nextCnt = static_cast<uint8_t>(v.cardsInCurrentTrick + 1U);

        HandMask normalLegal = legal & ~kJokerFollowActionsMask;
        while (normalLegal != 0ULL) {
            const CardId cid = LowestCardInMask(normalLegal);
            normalLegal &= (normalLegal - 1ULL);
            simTrick[v.cardsInCurrentTrick] = TrickCard{
                me, CardPlay{.cardId = cid, .jokerAction = JokerAction::None, .requestedSuit = Suit::None}
            };
            if (rules::DetermineTrickWinnerN(simTrick, nextCnt, v.trumpSuit) == me) {
                ++winCards;
            } else {
                ++loseCards;
            }
        }

        const int myJokers = std::popcount(v.myHand & kJokerMask);
        if (myJokers > 0) {
            if (((legal >> kActionJokerPlayWin) & 1ULL) != 0ULL) winCards += myJokers;
            if (((legal >> kActionJokerPlayLow) & 1ULL) != 0ULL) loseCards += myJokers;
        }

        const float denom = static_cast<float>(std::max(1, handSize));
        out[318] = std::min(1.0f, static_cast<float>(winCards) / denom);
        out[319] = std::min(1.0f, static_cast<float>(loseCards) / denom);
    }
}

} // namespace

void GameStateView::WriteSoloTensor(float* out) const noexcept {
    std::fill_n(out, kTensorSize, 0.0f);

    const HorizonInfo h = ComputeHorizon(*this);
    WriteBlockA(*this, h, out);

    const PlayerId me = observerPlayer;
    const float myScore = static_cast<float>(totalScores[me]);
    const float matchPot = std::max(100.0f, static_cast<float>(h.remainingMatchCards) * 25.0f);

    int strictlyBetter = 0;
    for (int rel = 1; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, me);
        const float diff = static_cast<float>(totalScores[absP]) - myScore;
        if (totalScores[absP] > totalScores[me]) {
            ++strictlyBetter;
        }
        out[18 + (rel - 1)] = diff / 500.0f;
        out[21 + (rel - 1)] = diff / matchPot;
    }
    out[24] = static_cast<float>(3 - strictlyBetter) / 3.0f;

    const auto baseDeltas = ComputePulkaDeltas(*this, pulkaClean);
    WriteBlockBPulkaCommon(*this, baseDeltas, out);

    WriteBlockC(*this, h, out);
    WriteBlockD(*this, out);
}

void GameStateView::WritePairsTensor(float* out) const noexcept {
    std::fill_n(out, kTensorSize, 0.0f);

    const HorizonInfo h = ComputeHorizon(*this);
    WriteBlockA(*this, h, out);

    const PlayerId me = observerPlayer;
    const PlayerId enemyUnder = AbsSeat(1, me);
    const PlayerId partner = AbsSeat(2, me);
    const PlayerId enemyOver = AbsSeat(3, me);

    const float myTeamScore = static_cast<float>(totalScores[me] + totalScores[partner]);
    const float enemyTeamScore = static_cast<float>(totalScores[enemyOver] + totalScores[enemyUnder]);
    const float diff = enemyTeamScore - myTeamScore;
    const float matchPot = std::max(100.0f, static_cast<float>(h.remainingMatchCards) * 50.0f);

    out[18] = diff / 500.0f;
    out[19] = diff / matchPot;
    out[20] = (myTeamScore > enemyTeamScore) ? 1.0f : 0.0f;

    const int myTeamClean = (pulkaClean[me] ? 1 : 0) + (pulkaClean[partner] ? 1 : 0);
    const int enemyTeamClean = (pulkaClean[enemyOver] ? 1 : 0) + (pulkaClean[enemyUnder] ? 1 : 0);
    out[21] = static_cast<float>(myTeamClean) / 2.0f;
    out[22] = static_cast<float>(enemyTeamClean) / 2.0f;

    const auto baseDeltas = ComputePulkaDeltas(*this, pulkaClean);
    out[23] = (baseDeltas[me] + baseDeltas[partner]) / 500.0f;
    out[24] = (baseDeltas[enemyOver] + baseDeltas[enemyUnder]) / 500.0f;

    WriteBlockBPulkaCommon(*this, baseDeltas, out);

    WriteBlockC(*this, h, out);
    WriteBlockD(*this, out);
}

void GameStateView::WriteTensor(float* outBuffer) const noexcept {
    if (teamMode == TeamMode::Pairs) {
        WritePairsTensor(outBuffer);
    } else {
        WriteSoloTensor(outBuffer);
    }
}

} // namespace joker::core