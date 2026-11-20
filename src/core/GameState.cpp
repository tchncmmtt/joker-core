#include "core/GameState.hpp"
#include "core/Rules.hpp"
#include <algorithm>
#include <bit>
#include <utility>

namespace joker::core {

namespace {

constexpr int RelSeat(const PlayerId absSeat, const PlayerId observer) noexcept {
    return (static_cast<int>(absSeat) - static_cast<int>(observer) + kNumPlayers) % kNumPlayers;
}

constexpr PlayerId AbsSeat(const int relSeat, const PlayerId observer) noexcept {
    return static_cast<PlayerId>((static_cast<int>(observer) + relSeat) % kNumPlayers);
}

constexpr float CalcRawBonusValue(const GameStateView& v, const PlayerId absSeat) noexcept {
    if (!v.pulkaClean[absSeat]) {
        return 0.0f;
    }
    return v.pulkaAllPasses[absSeat]
        ? 450.0f
        : static_cast<float>(v.pulkaMaxScoreExclLast[absSeat]);
}

constexpr float CalcRawPenaltyValue(const GameStateView& v, const PlayerId absSeat) noexcept {
    if (v.pulkaClean[absSeat]) {
        return 0.0f;
    }
    return static_cast<float>(v.pulkaMaxNonPassScoreExclLast[absSeat]);
}

    // Эффективный ожидаемый штраф с учётом PulkaMode (Strict vs Soft)
constexpr float CalcEffectivePenaltyRisk(const GameStateView& v, const PlayerId absSeat) noexcept {
    if (v.pulkaClean[absSeat]) {
        return 0.0f;
    }
    if (v.pulkaMode == PulkaMode::Soft) {
        // В мягком режиме штраф списывается только если вышел игрок "над" нами (prev = (absSeat + 3) % 4)
        const PlayerId over = static_cast<PlayerId>((absSeat + kNumPlayers - 1) % kNumPlayers);
        return v.pulkaClean[over] ? CalcRawPenaltyValue(v, absSeat) : 0.0f;
    }
    return CalcRawPenaltyValue(v, absSeat);
}

constexpr PlayerId DeterminePartialTrickWinner(
    const std::array<TrickCard, kNumPlayers>& trick,
    const uint8_t count,
    const Suit trumpSuit) noexcept {
    [[assume(count > 0 && count < kNumPlayers)]];

    for (int i = static_cast<int>(count) - 1; i >= 0; --i) {
        const auto& play = trick[i].play;
        if (GetCard(play.cardId).IsJoker() && play.jokerAction == JokerAction::PlayWin) {
            return trick[i].player;
        }
    }

    const auto& [cardId, jokerAction, requestedSuit] = trick[0].play;
    const Card& leadCard = GetCard(cardId);

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

    const Suit effectiveLeadSuit = leadCard.IsJoker() ? requestedSuit : leadCard.suit;
    PlayerId winner = trick[0].player;
    Suit winningSuit = leadCard.IsJoker() ? Suit::None : leadCard.suit;
    Rank winningRank = leadCard.rank;
    bool hasValidCardWinner = !leadCard.IsJoker();

    for (uint8_t i = 1; i < count; ++i) {
        const Card& c = GetCard(trick[i].play.cardId);
        if (c.IsJoker()) continue;

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

inline void WriteCardMask(float* dst, const HandMask mask) noexcept {
    for (size_t id = 0; id < kMaxDeckSize; ++id) {
        dst[id] = ((mask & (1ULL << id)) != 0) ? 1.0f : 0.0f;
    }
}

// Общий блок 0..15: Фаза, универсальный горизонт пулек и позиция дилера
// (Флаги RulePreset и MatchMode исключены, так как под каждый режим обучается отдельная сеть)
inline void WriteHeaderBlock(const GameStateView& v, float* out) noexcept {
    if (v.phase != Phase::MatchFinished) {
        out[std::to_underlying(v.phase)] = 1.0f; // 0..2: Phase one-hot
    }

    const int totalDeals = GetTotalDeals(v.mode);
    const int totalPulkas = GetPulkasCount(v.mode);
    const auto& sched = GetScheduleEntry(v.mode, std::clamp(v.dealNumber, 0, totalDeals - 1));
    const int pulkaLen = (v.mode == MatchMode::Short || sched.isNineCardDeal) ? 4 : 8;
    const int remainingPulkas = std::max(0, totalPulkas - 1 - v.pulkaIndex);
    const int remainingDealsInPulka = std::max(0, pulkaLen - 1 - v.roundInPulka);

    out[3] = static_cast<float>(remainingPulkas) / 4.0f;       // 0.0 на финальной пульке матча (и в Long, и в Short)
    out[4] = static_cast<float>(v.roundInPulka) / static_cast<float>(pulkaLen);
    out[5] = static_cast<float>(remainingDealsInPulka) / 8.0f; // сколько сдач осталось в текущей пульке
    out[6] = static_cast<float>(v.cardsInDeal) / 9.0f;
    out[7] = sched.isPulkaEnd ? 1.0f : 0.0f;

    const int relDealer = RelSeat(v.dealerPlayer, v.observerPlayer);
    out[8 + relDealer] = 1.0f; // 8..11: относительный дилер one-hot
    // 12..15: резерв (0.0)
}

// Общий блок 120..383: Карточные матрицы (6 * 38) и память стола (ренонсы + счётчики)
inline void WriteCardsAndMemoryBlock(const GameStateView& v, float* out) noexcept {
    WriteCardMask(out + 120, v.myHand);
    WriteCardMask(out + 158, v.playedCardsHistory);

    const HandMask upcardMask = (v.upcard != kInvalidCard && v.upcard < kMaxDeckSize)
        ? (1ULL << v.upcard) : 0ULL;
    WriteCardMask(out + 196, upcardMask);

    for (uint8_t i = 0; i < 3; ++i) {
        const HandMask trickCardMask = (i < v.cardsInCurrentTrick)
            ? (1ULL << v.currentTrick[i].play.cardId) : 0ULL;
        WriteCardMask(out + 234 + i * kMaxDeckSize, trickCardMask);
    }

    for (int rel = 1; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, v.observerPlayer);
        const uint8_t voids = v.voidSuits[absP];
        const int base = 348 + (rel - 1) * 4;
        for (int s = 0; s < 4; ++s) {
            out[base + s] = ((voids & (1U << s)) != 0) ? 1.0f : 0.0f;
        }
    }

    HandMask visibleMask = v.myHand | v.playedCardsHistory | upcardMask;
    for (uint8_t i = 0; i < v.cardsInCurrentTrick; ++i) {
        visibleMask |= (1ULL << v.currentTrick[i].play.cardId);
    }
    const HandMask fullDeck = rules::GetFullDeckMask(v.rules);
    const HandMask unseenMask = fullDeck & ~visibleMask;

    const float suitCap = (v.rules == RulePreset::Classic) ? 9.0f : 8.0f;
    for (int s = 0; s < 4; ++s) {
        const float maxSuitCards = (s >= 2) ? 9.0f : suitCap;
        const HandMask sMask = GetSuitMask(static_cast<Suit>(s));
        out[360 + s] = static_cast<float>(std::popcount(unseenMask & sMask)) / maxSuitCards;
        out[365 + s] = static_cast<float>(std::popcount(v.myHand & sMask)) / maxSuitCards;
    }
    out[364] = static_cast<float>(std::popcount(unseenMask & kJokerMask)) / 2.0f;
    out[369] = static_cast<float>(std::popcount(v.myHand & kJokerMask)) / 2.0f;
}

} // namespace

void GameState::ResetForNewMatch(const RulePreset preset,
                                 const MatchMode matchMode,
                                 const TeamMode tMode,
                                 const PulkaMode pMode) noexcept {
    hands.fill(0);
    playedCardsHistory = 0;
    totalScores.fill(0);
    ResetPulkaTrackers();

    currentTrick.fill(TrickCard{});
    bids.fill(-1);
    tricksTaken.fill(0);
    voidSuits.fill(0);
    jokersPlayedBy.fill(0);

    dealNumber = 0;
    cardsInDeal = (matchMode == MatchMode::Short) ? 9 : 1;
    bidsMadeInDeal = 0;
    tricksCompletedInDeal = 0;

    rules = preset;
    mode = matchMode;
    teamMode = tMode;
    pulkaMode = pMode;
    phase = Phase::ChoosingTrump;
    dealer = 0;
    currentPlayer = 1;
    trickLeader = 1;
    lastTrickWinner = 0;
    trumpSuit = Suit::None;
    upcard = kInvalidCard;
    cardsInCurrentTrick = 0;
}

void GameState::ResetPulkaTrackers() noexcept {
    pulkaMaxScoreExclLast.fill(0);
    pulkaMaxNonPassScoreExclLast.fill(0);
    pulkaClean.fill(true);
    pulkaAllPasses.fill(true);
}

GameStateView GameState::CreateView(const PlayerId observer) const noexcept {
    const int totalDeals = GetTotalDeals(mode);
    const auto& sched = GetScheduleEntry(mode, std::clamp(static_cast<int>(dealNumber), 0, totalDeals - 1));

    GameStateView view{};
    view.rules = rules;
    view.mode = mode;
    view.teamMode = teamMode;
    view.pulkaMode = pulkaMode;
    view.phase = phase;
    view.observerPlayer = observer;
    view.dealerPlayer = dealer;
    view.currentPlayer = currentPlayer;

    view.dealNumber = dealNumber;
    view.pulkaIndex = sched.pulkaIndex;
    view.roundInPulka = sched.roundInPulka;
    view.cardsInDeal = cardsInDeal;

    view.trumpSuit = trumpSuit;
    view.upcard = upcard;

    view.myHand = hands[observer];
    view.playedCardsHistory = playedCardsHistory;

    view.bids = bids;
    for (int i = 0; i < kNumPlayers; ++i) {
        view.tricksTaken[i] = tricksTaken[i];
        view.totalScores[i] = totalScores[i];
        view.pulkaMaxScoreExclLast[i] = pulkaMaxScoreExclLast[i];
        view.pulkaMaxNonPassScoreExclLast[i] = pulkaMaxNonPassScoreExclLast[i];
    }

    view.pulkaClean = pulkaClean;
    view.pulkaAllPasses = pulkaAllPasses;
    view.voidSuits = voidSuits;
    view.jokersPlayedBy = jokersPlayedBy;

    view.currentTrick = currentTrick;
    view.cardsInCurrentTrick = cardsInCurrentTrick;

    return view;
}

void GameStateView::WriteSoloTensor(float* out) const noexcept {
    std::fill_n(out, kTensorSize, 0.0f);

    WriteHeaderBlock(*this, out);

    // 16..39: очки и экономика пульки (Solo)
    const PlayerId me = observerPlayer;
    const float myScore = static_cast<float>(totalScores[me]);
    int cleanCount = 0;

    for (int rel = 0; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, me);
        const float pScore = static_cast<float>(totalScores[absP]);

        out[16 + rel] = pScore / 1000.0f;
        if (rel > 0) {
            out[20 + (rel - 1)] = (pScore - myScore) / 1000.0f;
        }

        const bool clean = pulkaClean[absP];
        if (clean) ++cleanCount;
        out[23 + rel] = clean ? 1.0f : 0.0f;
        out[27 + rel] = CalcRawBonusValue(*this, absP) / 500.0f;
        out[31 + rel] = CalcEffectivePenaltyRisk(*this, absP) / 500.0f;
    }
    out[35] = static_cast<float>(cleanCount) / 4.0f;

    // 40..67: контракты сдачи
    for (int rel = 0; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, me);
        const int b = bids[absP];
        const int t = tricksTaken[absP];
        const bool hasB = (b >= 0);

        out[40 + rel] = hasB ? (static_cast<float>(b) / 9.0f) : 0.0f;
        out[44 + rel] = hasB ? 1.0f : 0.0f;
        out[48 + rel] = static_cast<float>(t) / 9.0f;
        out[52 + rel] = hasB ? (static_cast<float>(std::max(0, b - t)) / 9.0f) : 0.0f;
        out[56 + rel] = (hasB && t == b) ? 1.0f : 0.0f;
        out[60 + rel] = (hasB && t > b) ? 1.0f : 0.0f;
        out[64 + rel] = (hasB && b > 0 && t == 0) ? 1.0f : 0.0f;
    }

    // 68..119: механика хода
    if (phase == Phase::Bidding && me == dealerPlayer) {
        int sumOther = 0;
        for (int p = 0; p < kNumPlayers; ++p) {
            if (p != me && bids[p] >= 0) sumOther += bids[p];
        }
        const int forbidden = cardsInDeal - sumOther;
        if (forbidden >= 0 && forbidden <= cardsInDeal) {
            out[68 + forbidden] = 1.0f;
        }
    }

    out[78 + std::to_underlying(trumpSuit)] = 1.0f;

    Suit reqSuit = Suit::None;
    bool forceHigh = false;
    if (phase == Phase::TrickPlay && cardsInCurrentTrick > 0) {
        const auto& leadPlay = currentTrick[0].play;
        const Card& leadCard = GetCard(leadPlay.cardId);
        reqSuit = leadCard.IsJoker() ? leadPlay.requestedSuit : leadCard.suit;
        forceHigh = leadCard.IsJoker() && (leadPlay.jokerAction == JokerAction::LeadHighSuit);
    }
    out[83 + std::to_underlying(reqSuit)] = 1.0f;

    out[88] = static_cast<float>(cardsInCurrentTrick) / 3.0f;
    out[89] = forceHigh ? 1.0f : 0.0f;
    out[90] = (phase == Phase::TrickPlay && cardsInCurrentTrick == 0) ? 1.0f : 0.0f;

    if (phase == Phase::TrickPlay && cardsInCurrentTrick > 0) {
        const PlayerId curWinner = DeterminePartialTrickWinner(currentTrick, cardsInCurrentTrick, trumpSuit);
        out[91 + RelSeat(curWinner, me)] = 1.0f;
    }

    for (uint8_t i = 0; i < cardsInCurrentTrick && i < 3; ++i) {
        const int base = 95 + i * 8;
        out[base + RelSeat(currentTrick[i].player, me)] = 1.0f;
        const JokerAction ja = currentTrick[i].play.jokerAction;
        if (ja != JokerAction::None) {
            out[base + 4 + (std::to_underlying(ja) - 1)] = 1.0f;
        }
    }

    WriteCardsAndMemoryBlock(*this, out);
}

void GameStateView::WritePairsTensor(float* out) const noexcept {
    std::fill_n(out, kTensorSize, 0.0f);

    WriteHeaderBlock(*this, out);

    // 16..39: командный счёт и пулька Pairs (0 = Я, 1 = Враг под нами, 2 = Партнёр, 3 = Враг над нами)
    const PlayerId me = observerPlayer;
    const PlayerId enemyUnder = AbsSeat(1, me);
    const PlayerId partner = AbsSeat(2, me);
    const PlayerId enemyOver = AbsSeat(3, me);

    const float myTeamScore = static_cast<float>(totalScores[me] + totalScores[partner]);
    const float enemyTeamScore = static_cast<float>(totalScores[enemyOver] + totalScores[enemyUnder]);

    out[16] = myTeamScore / 1000.0f;
    out[17] = enemyTeamScore / 1000.0f;
    out[18] = (myTeamScore - enemyTeamScore) / 1000.0f;

    for (int rel = 0; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, me);
        out[19 + rel] = pulkaClean[absP] ? 1.0f : 0.0f;
        out[23 + rel] = CalcRawBonusValue(*this, absP) / 500.0f;
        out[27 + rel] = CalcEffectivePenaltyRisk(*this, absP) / 500.0f;
    }

    const int myTeamClean = (pulkaClean[me] ? 1 : 0) + (pulkaClean[partner] ? 1 : 0);
    const int enemyTeamClean = (pulkaClean[enemyOver] ? 1 : 0) + (pulkaClean[enemyUnder] ? 1 : 0);

    float myTeamBonusSum = 0.0f;
    float myTeamPenaltyRisk = 0.0f;

    if (pulkaMode == PulkaMode::Soft) {
        // В мягком режиме каждый из пары получает премию, если не вышел соперник "над ним"
        if (pulkaClean[me] && !pulkaClean[enemyOver]) {
            myTeamBonusSum += CalcRawBonusValue(*this, me);
        }
        if (pulkaClean[partner] && !pulkaClean[enemyUnder]) {
            myTeamBonusSum += CalcRawBonusValue(*this, partner);
        }
        myTeamPenaltyRisk = CalcEffectivePenaltyRisk(*this, me) + CalcEffectivePenaltyRisk(*this, partner);
    } else {
        myTeamBonusSum = CalcRawBonusValue(*this, me) + CalcRawBonusValue(*this, partner);
        myTeamPenaltyRisk = (myTeamClean == 0)
            ? (CalcRawPenaltyValue(*this, me) + CalcRawPenaltyValue(*this, partner))
            : 0.0f;
    }

    out[31] = static_cast<float>(myTeamClean) / 2.0f;
    out[32] = static_cast<float>(enemyTeamClean) / 2.0f;
    out[33] = myTeamBonusSum / 500.0f;
    out[34] = myTeamPenaltyRisk / 500.0f;

    // 40..71: личные и командные контракты
    int totalBids = 0;
    for (int rel = 0; rel < kNumPlayers; ++rel) {
        const PlayerId absP = AbsSeat(rel, me);
        const int b = bids[absP];
        const int t = tricksTaken[absP];
        const bool hasB = (b >= 0);
        if (hasB) totalBids += b;

        out[40 + rel] = hasB ? (static_cast<float>(b) / 9.0f) : 0.0f;
        out[44 + rel] = hasB ? 1.0f : 0.0f;
        out[48 + rel] = static_cast<float>(t) / 9.0f;
        out[52 + rel] = hasB ? (static_cast<float>(std::max(0, b - t)) / 9.0f) : 0.0f;
        out[56 + rel] = (hasB && t == b) ? 1.0f : 0.0f;
        out[60 + rel] = (hasB && t > b) ? 1.0f : 0.0f;
        out[64 + rel] = (hasB && b > 0 && t == 0) ? 1.0f : 0.0f;
    }

    const int myBid = std::max<int>(0, bids[me]);
    const int partnerBid = std::max<int>(0, bids[partner]);
    const int enemy1Bid = std::max<int>(0, bids[enemyOver]);
    const int enemy2Bid = std::max<int>(0, bids[enemyUnder]);

    const int myNeed = (bids[me] >= 0) ? std::max(0, bids[me] - tricksTaken[me]) : 0;
    const int partnerNeed = (bids[partner] >= 0) ? std::max(0, bids[partner] - tricksTaken[partner]) : 0;

    out[68] = static_cast<float>(myBid + partnerBid) / 18.0f;
    out[69] = static_cast<float>(myNeed + partnerNeed) / 18.0f;
    out[70] = static_cast<float>(enemy1Bid + enemy2Bid) / 18.0f;
    out[71] = static_cast<float>(cardsInDeal - totalBids) / 9.0f;

    // 72..119: механика хода и контроль партнёра
    if (phase == Phase::Bidding && me == dealerPlayer) {
        int sumOther = 0;
        for (int p = 0; p < kNumPlayers; ++p) {
            if (p != me && bids[p] >= 0) sumOther += bids[p];
        }
        const int forbidden = cardsInDeal - sumOther;
        if (forbidden >= 0 && forbidden <= cardsInDeal) {
            out[72 + forbidden] = 1.0f;
        }
    }

    out[82 + std::to_underlying(trumpSuit)] = 1.0f;

    Suit reqSuit = Suit::None;
    bool forceHigh = false;
    if (phase == Phase::TrickPlay && cardsInCurrentTrick > 0) {
        const auto& leadPlay = currentTrick[0].play;
        const Card& leadCard = GetCard(leadPlay.cardId);
        reqSuit = leadCard.IsJoker() ? leadPlay.requestedSuit : leadCard.suit;
        forceHigh = leadCard.IsJoker() && (leadPlay.jokerAction == JokerAction::LeadHighSuit);
    }
    out[87 + std::to_underlying(reqSuit)] = 1.0f;

    out[92] = static_cast<float>(cardsInCurrentTrick) / 3.0f;
    out[93] = forceHigh ? 1.0f : 0.0f;
    out[94] = (phase == Phase::TrickPlay && cardsInCurrentTrick == 0) ? 1.0f : 0.0f;

    if (phase == Phase::TrickPlay && cardsInCurrentTrick > 0) {
        const PlayerId curWinner = DeterminePartialTrickWinner(currentTrick, cardsInCurrentTrick, trumpSuit);
        out[95] = (curWinner == partner) ? 1.0f : 0.0f;
    }

    for (uint8_t i = 0; i < cardsInCurrentTrick && i < 3; ++i) {
        const int base = 96 + i * 8;
        out[base + RelSeat(currentTrick[i].player, me)] = 1.0f;
        const JokerAction ja = currentTrick[i].play.jokerAction;
        if (ja != JokerAction::None) {
            out[base + 4 + (std::to_underlying(ja) - 1)] = 1.0f;
        }
    }

    WriteCardsAndMemoryBlock(*this, out);
}

void GameStateView::WriteTensor(float* outBuffer) const noexcept {
    if (teamMode == TeamMode::Pairs) {
        WritePairsTensor(outBuffer);
    } else {
        WriteSoloTensor(outBuffer);
    }
}

} // namespace joker::core