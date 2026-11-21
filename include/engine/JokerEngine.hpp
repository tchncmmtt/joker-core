#pragma once

#include "core/Deck.hpp"
#include "core/GameState.hpp"
#include "core/Rules.hpp"
#include <array>
#include <span>
#include <type_traits>

namespace joker::engine {

class JokerEngine {
public:
    explicit JokerEngine(
        core::RulePreset preset = core::RulePreset::Popular,
        uint64_t seed = 42,
        core::MatchMode mode = core::MatchMode::Long,
        core::TeamMode teamMode = core::TeamMode::Solo,
        core::PulkaMode pulkaMode = core::PulkaMode::Strict) noexcept;

    void Reset(core::RulePreset preset,
               uint64_t seed,
               core::MatchMode mode = core::MatchMode::Long,
               core::TeamMode teamMode = core::TeamMode::Solo,
               core::PulkaMode pulkaMode = core::PulkaMode::Strict) noexcept;

    void Reset256(core::RulePreset preset,
                  std::span<const uint64_t, 4> seed,
                  core::MatchMode mode = core::MatchMode::Long,
                  core::TeamMode teamMode = core::TeamMode::Solo,
                  core::PulkaMode pulkaMode = core::PulkaMode::Strict) noexcept;

    // Режим живого стола (Observed / Live Table Mode)
    void BeginObservedMatch(core::RulePreset preset,
                            core::MatchMode mode,
                            core::TeamMode teamMode,
                            core::PulkaMode pulkaMode,
                            core::PlayerId dealer,
                            core::PlayerId heroSeat) noexcept;

    // Старт режима живого стола с произвольной сдачи (вход в середину чужой партии)
    [[nodiscard]] bool SetObservedMatchState(
        core::RulePreset preset,
        core::MatchMode mode,
        core::TeamMode teamMode,
        core::PulkaMode pulkaMode,
        int dealNumber,
        core::PlayerId dealer,
        core::PlayerId heroSeat,
        std::span<const int32_t, core::kNumPlayers> totalScores,
        std::span<const uint8_t, core::kNumPlayers> pulkaClean,
        std::span<const uint8_t, core::kNumPlayers> pulkaAllPasses,
        std::span<const int16_t, core::kNumPlayers> pulkaMaxScoreExclLast,
        std::span<const int16_t, core::kNumPlayers> pulkaMaxNonPassScoreExclLast) noexcept;

    [[nodiscard]] bool SetObservedDeal(core::HandMask heroHandMask,
                                       core::CardId upcardOrInvalid,
                                       uint8_t trumpSuitOrNone) noexcept;

    [[nodiscard]] bool SetObservedRemainingSix(core::HandMask fullNineHandMask) noexcept;

    void ForceDeckOrder(std::span<const core::CardId, core::kMaxDeckSize> customCards) noexcept {
        deck_.ForceDeckOrder(customCards);
    }

    void Determinize(core::PlayerId observer, uint64_t rngSeed) noexcept;

    [[nodiscard]] bool IsDone() const noexcept { return state_.phase == core::Phase::MatchFinished; }
    [[nodiscard]] core::Phase GetPhase() const noexcept { return state_.phase; }
    [[nodiscard]] core::PlayerId GetCurrentPlayer() const noexcept { return state_.currentPlayer; }
    [[nodiscard]] core::ActionMask GetLegalActionsMask() const noexcept;

    [[nodiscard]] core::StepError ValidateAction(core::ActionId action) const noexcept {
        return core::rules::ValidateActionForState(state_, action);
    }

    bool Step(core::ActionId action) noexcept;
    core::StepError StepChecked(core::ActionId action) noexcept;

    [[nodiscard]] const core::GameState& GetState() const noexcept { return state_; }
    [[nodiscard]] std::array<int32_t, 2> GetTeamScores() const noexcept {
        return {
            state_.totalScores[0] + state_.totalScores[2],
            state_.totalScores[1] + state_.totalScores[3]
        };
    }

    [[nodiscard]] core::GameStateView GetObservation(core::PlayerId observer) const noexcept {
        return state_.CreateView(observer);
    }
    void WriteObservationTensor(core::PlayerId observer, float* outBuffer) const noexcept {
        state_.CreateView(observer).WriteTensor(outBuffer);
    }

private:
    void BeginDeal(int dealIdx) noexcept;
    void FinalizeDealScoring() noexcept;
    void ApplyPulkaBonuses() noexcept;
    void ApplySoloStrictPulkaBonuses() noexcept;
    void ApplyPairsStrictPulkaBonuses() noexcept;
    void ApplySoftPulkaBonuses() noexcept;

    core::Deck deck_;
    core::GameState state_{};
};

static_assert(std::is_trivially_copyable_v<JokerEngine>,
              "JokerEngine must be trivially copyable for fast MCTS cloning!");

} // namespace joker::engine