#pragma once

#include "engine/JokerEngine.hpp"
#include "joker_c_api.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace joker::engine {

struct RewardConfig {
    float scoreScale{0.01f};
    float relativeWeight{0.5f};
    float matchWinBonus{1.0f};
    float dealExactBonus{0.0f};
    float trickWinShaping{0.0f};
};

struct DealHistoryBuffer {
    static constexpr size_t kMaxSeqLen = 48;
    static constexpr size_t kTokenDim = 8;

    struct RawEvent {
        uint8_t eventType{0};
        uint8_t absPlayer{0};
        uint8_t actionId{0};
        uint8_t cardId{255};
        uint8_t suit{4};
        uint8_t jokerAction{0};
        uint8_t trickIdx{255};
        uint8_t posInTrick{255};
    };

    std::array<RawEvent, kMaxSeqLen> events{};
    uint8_t count{0};

    constexpr void Clear() noexcept {
        count = 0;
    }

    constexpr void Push(const RawEvent& ev) noexcept {
        if (count < kMaxSeqLen) {
            events[count++] = ev;
        }
    }

    [[nodiscard]] static RawEvent MakeEvent(
        const core::GameState& stBefore,
        core::ActionId action) noexcept;

    void WriteTokens(core::PlayerId observer, int16_t* outTokens, uint8_t* outLength) const noexcept;
};

struct alignas(64) EnvSlot {
    JokerEngine engine{};
    uint64_t episodeCounter{0};
    uint64_t rngState{0};
    DealHistoryBuffer history{};
};

struct alignas(64) SlotStats {
    std::array<JokerPlayerStats, core::kNumPlayers> players{};
    std::array<uint8_t, core::kNumPlayers> matchBonusesCount{};

    void Reset() noexcept;
};

class JokerVectorEnv {
public:
    static constexpr size_t kOracleTensorSize = 160;

    JokerVectorEnv(
        size_t numEnvs,
        core::RulePreset preset = core::RulePreset::Popular,
        uint64_t baseSeed = 42,
        int numThreads = 0,
        core::MatchMode mode = core::MatchMode::Long,
        core::TeamMode teamMode = core::TeamMode::Solo,
        core::PulkaMode pulkaMode = core::PulkaMode::Strict,
        RewardConfig rewardConfig = {});

    [[nodiscard]] size_t Size() const noexcept { return slots_.size(); }
    [[nodiscard]] int GetNumThreads() const noexcept { return numThreads_; }
    [[nodiscard]] core::RulePreset GetRulePreset() const noexcept { return preset_; }
    [[nodiscard]] core::MatchMode GetMatchMode() const noexcept { return mode_; }
    [[nodiscard]] core::TeamMode GetTeamMode() const noexcept { return teamMode_; }
    [[nodiscard]] core::PulkaMode GetPulkaMode() const noexcept { return pulkaMode_; }
    [[nodiscard]] const RewardConfig& GetRewardConfig() const noexcept { return rewardConfig_; }

    void SetRewardConfig(const RewardConfig& cfg) noexcept { rewardConfig_ = cfg; }
    void SetAutoReset(bool enabled) noexcept { autoReset_ = enabled; }
    [[nodiscard]] bool GetAutoReset() const noexcept { return autoReset_; }

    void SetStatsEnabled(bool enabled) noexcept { collectStats_ = enabled; }
    [[nodiscard]] bool GetStatsEnabled() const noexcept { return collectStats_; }
    void ResetStats() noexcept;
    void GetAggregatedStats(std::span<JokerPlayerStats, core::kNumPlayers> outStats) const noexcept;

    void Reset(
        core::RulePreset preset,
        uint64_t baseSeed,
        core::MatchMode mode = core::MatchMode::Long,
        core::TeamMode teamMode = core::TeamMode::Solo,
        core::PulkaMode pulkaMode = core::PulkaMode::Strict) noexcept;

    void ResetWithSeeds(
        core::RulePreset preset,
        core::MatchMode mode,
        core::TeamMode teamMode,
        core::PulkaMode pulkaMode,
        std::span<const uint64_t> seedsPerEnv) noexcept;

    void ForkAndDeterminize(
        const JokerEngine& rootEngine,
        core::PlayerId observer,
        uint64_t baseSeed,
        std::span<float> outObsBatch = {},
        std::span<uint64_t> outLegalMasks = {},
        std::span<float> outOracleBatch = {},
        const DealHistoryBuffer* rootHistory = nullptr) noexcept;

    void GetObservations(
        std::span<float> outObsBatch,
        std::span<uint64_t> outLegalMasks,
        std::span<uint8_t> outCurrentPlayers,
        std::span<float> outOracleBatch = {},
        std::span<int16_t> outHistoryTokens = {},
        std::span<uint8_t> outHistoryLengths = {}) const noexcept;

    void StepBatch(
        std::span<const uint8_t> actions,
        std::span<float> outNextObsBatch,
        std::span<uint64_t> outNextMasks,
        std::span<uint8_t> outNextPlayers,
        std::span<float> outRewards,
        std::span<uint8_t> outDealDones,
        std::span<uint8_t> outMatchDones,
        std::span<int32_t> outFinalScores,
        std::span<float> outNextOracleBatch = {},
        std::span<int16_t> outNextHistoryTokens = {},
        std::span<uint8_t> outNextHistoryLengths = {}) noexcept;

    void AdvanceToHero(
        std::span<const uint8_t> heroSeats,
        std::span<float> outObsBatch,
        std::span<uint64_t> outLegalMasks,
        std::span<float> outOracleBatch = {},
        std::span<int16_t> outHistoryTokens = {},
        std::span<uint8_t> outHistoryLengths = {}) noexcept;

    void StepArena(
        std::span<const uint8_t> heroActions,
        std::span<const uint8_t> heroSeats,
        std::span<float> outNextObsBatch,
        std::span<uint64_t> outNextMasks,
        std::span<float> outHeroRewards,
        std::span<float> outAllRewards,
        std::span<uint8_t> outDealDones,
        std::span<uint8_t> outMatchDones,
        std::span<int32_t> outFinalScores,
        std::span<float> outNextOracleBatch = {},
        std::span<int16_t> outNextHistoryTokens = {},
        std::span<uint8_t> outNextHistoryLengths = {}) noexcept;

    [[nodiscard]] static core::ActionId SelectHeuristicAction(
        const JokerEngine& env,
        uint64_t& rngState) noexcept;

private:
    void GenerateEnvSeed256(size_t envIdx, uint64_t episodeIdx, uint64_t outSeed[4]) const noexcept;

    void RecordStepStats(
        size_t envIdx,
        core::Phase prevPhase,
        core::PlayerId actor,
        core::ActionId action,
        int prevDeal,
        int prevCardsInDeal,
        const std::array<int8_t, core::kNumPlayers>& prevBids,
        const std::array<int8_t, core::kNumPlayers>& finalDealTricks,
        const std::array<int32_t, core::kNumPlayers>& prevScores,
        const std::array<int32_t, core::kNumPlayers>& afterScores,
        const std::array<bool, core::kNumPlayers>& prevPulkaClean,
        bool dealDone,
        bool matchDone) noexcept;

    void ComputeStepRewards(
        const std::array<int32_t, core::kNumPlayers>& prevScores,
        const std::array<int32_t, core::kNumPlayers>& afterScores,
        bool trickCompleted,
        core::PlayerId trickWinner,
        const std::array<int8_t, core::kNumPlayers>& bids,
        const std::array<int8_t, core::kNumPlayers>& tricksTaken,
        bool dealDone,
        bool matchDone,
        std::array<float, core::kNumPlayers>& outStepRewards) const noexcept;

    static void WriteOracleTensor(
        const core::GameState& state,
        core::PlayerId observer,
        float* outOracle) noexcept;

    std::vector<EnvSlot> slots_;
    std::vector<SlotStats> slotStats_;
    core::RulePreset preset_{core::RulePreset::Popular};
    core::MatchMode mode_{core::MatchMode::Long};
    core::TeamMode teamMode_{core::TeamMode::Solo};
    core::PulkaMode pulkaMode_{core::PulkaMode::Strict};
    RewardConfig rewardConfig_{};
    uint64_t baseSeed_{42};
    int numThreads_{1};
    bool autoReset_{true};
    bool collectStats_{false};
};

} // namespace joker::engine