#include "core/GameState.hpp"
#include <algorithm>

namespace joker::core {

void GameState::ResetForNewMatch(const RulePreset preset,
                                 const MatchMode matchMode,
                                 const TeamMode tMode,
                                 const PulkaMode pMode) noexcept {
    hands.fill(0);
    impossibleCards.fill(0);
    playedCardsHistory = 0;
    totalScores.fill(0);
    ResetPulkaTrackers();

    currentTrick.fill(TrickCard{});
    bids.fill(-1);
    tricksTaken.fill(0);
    handCounts.fill(0);
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
    knownHandsMask = 0x0F;
    observedFlags = 0;
    heroSeat = 0;
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
        view.impossibleCards[i] = impossibleCards[i];
        view.voidSuits[i] = ComputeVoidSuitsMask(static_cast<PlayerId>(i));
        view.handCounts[i] = handCounts[i];
    }

    view.pulkaClean = pulkaClean;
    view.pulkaAllPasses = pulkaAllPasses;
    view.jokersPlayedBy = jokersPlayedBy;

    view.currentTrick = currentTrick;
    view.cardsInCurrentTrick = cardsInCurrentTrick;

    return view;
}

} // namespace joker::core