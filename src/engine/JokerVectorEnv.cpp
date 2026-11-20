#include "engine/JokerVectorEnv.hpp"
#include <algorithm>
#include <bit>
#include <random>
#include <thread>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace joker::engine {

using namespace joker::core;

namespace {

constexpr uint8_t RelSeat1Based(const PlayerId absSeat, const PlayerId observer) noexcept {
    return static_cast<uint8_t>(((static_cast<int>(absSeat) - static_cast<int>(observer) + kNumPlayers) % kNumPlayers) + 1);
}

// Исполнение одного шага с записью события в историю сдачи
inline ActionId ExecuteAndRecordStep(EnvSlot& slot, ActionId requestedAction) noexcept {
    JokerEngine& env = slot.engine;
    const GameState& st = env.GetState();

    const Phase prevPhase = st.phase;
    const PlayerId actor = st.currentPlayer;
    const uint8_t trickIdx = static_cast<uint8_t>(st.tricksCompletedInDeal);
    const uint8_t posInTrick = st.cardsInCurrentTrick;
    const HandMask actorHand = st.hands[actor];

    const ActionMask mask = env.GetLegalActionsMask();
    ActionId act = requestedAction;
    if (act >= 64 || ((mask >> act) & 1ULL) == 0ULL) {
        act = (mask != 0ULL) ? static_cast<ActionId>(std::countr_zero(mask)) : 0;
    }

    DealHistoryBuffer::RawEvent ev{};
    ev.absPlayer = actor;
    ev.actionId = act;

    if (prevPhase == Phase::ChoosingTrump) {
        ev.eventType = 1;
        ev.suit = act; // 0..4
    } else if (prevPhase == Phase::Bidding) {
        ev.eventType = 2;
        ev.suit = 255;
    } else if (prevPhase == Phase::TrickPlay) {
        ev.eventType = 3;
        const CardPlay play = DecodeCardPlayAction(act, actorHand);
        ev.cardId = play.cardId;
        ev.jokerAction = std::to_underlying(play.jokerAction);
        const Card& c = GetCard(play.cardId);
        ev.suit = c.IsJoker() ? std::to_underlying(play.requestedSuit) : std::to_underlying(c.suit);
        ev.trickIdx = trickIdx;
        ev.posInTrick = posInTrick;
    }

    if (env.Step(act) && ev.eventType != 0) {
        slot.history.Push(ev);
    }

    return act;
}

} // namespace

void DealHistoryBuffer::WriteTokens(const PlayerId observer, int16_t* outTokens, uint8_t* outLength) const noexcept {
    constexpr size_t kTotalInts = kMaxSeqLen * kTokenDim;
    std::fill_n(outTokens, kTotalInts, static_cast<int16_t>(0));

    for (uint8_t i = 0; i < count; ++i) {
        const RawEvent& ev = events[i];
        int16_t* dst = outTokens + static_cast<size_t>(i) * kTokenDim;

        dst[0] = static_cast<int16_t>(ev.eventType);                          // 1..3 (0 = PAD)
        dst[1] = static_cast<int16_t>(RelSeat1Based(ev.absPlayer, observer)); // 1..4
        dst[2] = static_cast<int16_t>(ev.actionId + 1U);                      // 1..46
        dst[3] = (ev.cardId < kMaxDeckSize)
                     ? static_cast<int16_t>(ev.cardId + 1U)
                     : static_cast<int16_t>(0);                               // 1..38 или 0
        dst[4] = (ev.suit <= 4U)
                     ? static_cast<int16_t>(ev.suit + 1U)
                     : static_cast<int16_t>(0);                               // 1..5 или 0
        dst[5] = (ev.eventType == 3U)
                     ? static_cast<int16_t>(ev.jokerAction + 1U)
                     : static_cast<int16_t>(0);                               // 1..5 или 0
        dst[6] = (ev.trickIdx < 9U)
                     ? static_cast<int16_t>(ev.trickIdx + 1U)
                     : static_cast<int16_t>(0);                               // 1..9 или 0
        dst[7] = (ev.posInTrick < 4U)
                     ? static_cast<int16_t>(ev.posInTrick + 1U)
                     : static_cast<int16_t>(0);                               // 1..4 или 0
    }

    if (outLength) {
        *outLength = count;
    }
}

JokerVectorEnv::JokerVectorEnv(
    size_t numEnvs,
    RulePreset preset,
    uint64_t baseSeed,
    int numThreads,
    MatchMode mode,
    TeamMode teamMode,
    PulkaMode pulkaMode,
    RewardConfig rewardConfig)
    : slots_(numEnvs),
      preset_(preset),
      mode_(mode),
      teamMode_(teamMode),
      pulkaMode_(pulkaMode),
      rewardConfig_(rewardConfig),
      baseSeed_(baseSeed) {
    if (numThreads <= 0) {
        const unsigned int hw = std::thread::hardware_concurrency();
        numThreads_ = (hw > 0U) ? static_cast<int>(hw) : 4;
    } else {
        numThreads_ = numThreads;
    }

    Reset(preset, baseSeed, mode, teamMode, pulkaMode);
}

void JokerVectorEnv::GenerateEnvSeed256(
    size_t envIdx,
    uint64_t episodeIdx,
    uint64_t outSeed[4]) const noexcept {
    uint64_t z = baseSeed_
                 ^ (static_cast<uint64_t>(envIdx + 1ULL) * 0x9E3779B97F4A7C15ULL)
                 ^ (episodeIdx * 0xBF58476D1CE4E5B9ULL);
    for (int k = 0; k < 4; ++k) {
        outSeed[k] = Deck::SplitMix64(z);
    }
}

void JokerVectorEnv::Reset(
    RulePreset preset,
    uint64_t baseSeed,
    MatchMode mode,
    TeamMode teamMode,
    PulkaMode pulkaMode) noexcept {
    preset_ = preset;
    mode_ = mode;
    teamMode_ = teamMode;
    pulkaMode_ = pulkaMode;

    if (baseSeed == 0ULL) {
        std::random_device rd;
        baseSeed_ = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    } else {
        baseSeed_ = baseSeed;
    }

    const ptrdiff_t n = static_cast<ptrdiff_t>(slots_.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(numThreads_) if (n >= 16)
#endif
    for (ptrdiff_t i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        EnvSlot& slot = slots_[idx];
        slot.episodeCounter = 0ULL;
        uint64_t s256[4];
        GenerateEnvSeed256(idx, 0ULL, s256);
        slot.engine.Reset256(preset_, s256, mode_, teamMode_, pulkaMode_);
        slot.rngState = s256[0] ^ s256[2] ^ 0xD1B54A32D192ED03ULL;
        slot.history.Clear();
    }
}

void JokerVectorEnv::ForkAndDeterminize(
    const JokerEngine& rootEngine,
    const PlayerId observer,
    const uint64_t baseSeed,
    std::span<float> outObsBatch,
    std::span<uint64_t> outLegalMasks,
    std::span<float> outOracleBatch) noexcept {
    const ptrdiff_t n = static_cast<ptrdiff_t>(slots_.size());
    constexpr size_t kObsStride = GameStateView::kTensorSize;
    const PlayerId obsSeat = static_cast<PlayerId>(observer % kNumPlayers);

    // Синхронизируем режимы пула с клонируемой партией
    preset_ = rootEngine.GetState().rules;
    mode_ = rootEngine.GetState().mode;
    teamMode_ = rootEngine.GetState().teamMode;
    pulkaMode_ = rootEngine.GetState().pulkaMode;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(numThreads_) if (n >= 16)
#endif
    for (ptrdiff_t i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        EnvSlot& slot = slots_[idx];
        slot.engine = rootEngine;
        slot.episodeCounter = 0ULL;
        slot.history.Clear();

        uint64_t z = baseSeed ^ (static_cast<uint64_t>(idx + 1ULL) * 0x9E3779B97F4A7C15ULL);
        const uint64_t detSeed = Deck::SplitMix64(z);
        slot.rngState = Deck::SplitMix64(z);

        slot.engine.Determinize(obsSeat, detSeed);

        if (!outObsBatch.empty()) {
            slot.engine.WriteObservationTensor(obsSeat, outObsBatch.data() + idx * kObsStride);
        }
        if (!outLegalMasks.empty()) {
            outLegalMasks[idx] = slot.engine.GetLegalActionsMask();
        }
        if (!outOracleBatch.empty()) {
            WriteOracleTensor(slot.engine.GetState(), obsSeat, outOracleBatch.data() + idx * kOracleTensorSize);
        }
    }
}

void JokerVectorEnv::WriteOracleTensor(
    const GameState& state,
    const PlayerId observer,
    float* outOracle) noexcept {
    std::fill_n(outOracle, kOracleTensorSize, 0.0f);
    for (int rel = 0; rel < kNumPlayers; ++rel) {
        const PlayerId absP = static_cast<PlayerId>((static_cast<int>(observer) + rel) % kNumPlayers);
        const HandMask h = state.hands[absP];
        float* dst = outOracle + static_cast<size_t>(rel) * kMaxDeckSize;
        for (size_t id = 0; id < kMaxDeckSize; ++id) {
            dst[id] = ((h & (1ULL << id)) != 0ULL) ? 1.0f : 0.0f;
        }
    }
}

void JokerVectorEnv::ComputeStepRewards(
    const std::array<int32_t, kNumPlayers>& prevScores,
    const std::array<int32_t, kNumPlayers>& afterScores,
    const bool trickCompleted,
    const PlayerId trickWinner,
    const std::array<int8_t, kNumPlayers>& bids,
    const std::array<int8_t, kNumPlayers>& tricksTaken,
    const bool dealDone,
    const bool matchDone,
    std::array<float, kNumPlayers>& outStepRewards) const noexcept {
    std::array<float, kNumPlayers> rawDelta{};
    float sumDelta = 0.0f;
    for (int p = 0; p < kNumPlayers; ++p) {
        rawDelta[p] = static_cast<float>(afterScores[p] - prevScores[p]);
        sumDelta += rawDelta[p];
    }

    if (teamMode_ == TeamMode::Pairs) {
        const float teamADelta = rawDelta[0] + rawDelta[2];
        const float teamBDelta = rawDelta[1] + rawDelta[3];
        const float rA = 0.5f * (teamADelta - teamBDelta) * rewardConfig_.scoreScale;
        outStepRewards[0] = rA;
        outStepRewards[2] = rA;
        outStepRewards[1] = -rA;
        outStepRewards[3] = -rA;
    } else {
        const float w = rewardConfig_.relativeWeight;
        for (int p = 0; p < kNumPlayers; ++p) {
            const float oppMean = (sumDelta - rawDelta[p]) / 3.0f;
            outStepRewards[p] = (rawDelta[p] - w * oppMean) * rewardConfig_.scoreScale;
        }
    }

    if (trickCompleted && rewardConfig_.trickWinShaping != 0.0f && bids[trickWinner] >= 0) {
        const float sign = (tricksTaken[trickWinner] <= bids[trickWinner]) ? 1.0f : -1.0f;
        const float s = sign * rewardConfig_.trickWinShaping;
        if (teamMode_ == TeamMode::Pairs) {
            const int partner = (trickWinner + 2) % kNumPlayers;
            for (int p = 0; p < kNumPlayers; ++p) {
                outStepRewards[p] += (p == trickWinner || p == partner) ? s : -s;
            }
        } else {
            outStepRewards[trickWinner] += s;
        }
    }

    if (dealDone && rewardConfig_.dealExactBonus != 0.0f) {
        std::array<float, kNumPlayers> exactBonus{};
        for (int p = 0; p < kNumPlayers; ++p) {
            if (bids[p] >= 0 && tricksTaken[p] == bids[p]) {
                exactBonus[p] = rewardConfig_.dealExactBonus;
            }
        }
        if (teamMode_ == TeamMode::Pairs) {
            const float diff = 0.5f * ((exactBonus[0] + exactBonus[2]) - (exactBonus[1] + exactBonus[3]));
            outStepRewards[0] += diff;
            outStepRewards[2] += diff;
            outStepRewards[1] -= diff;
            outStepRewards[3] -= diff;
        } else {
            for (int p = 0; p < kNumPlayers; ++p) {
                outStepRewards[p] += exactBonus[p];
            }
        }
    }

    if (matchDone && rewardConfig_.matchWinBonus != 0.0f) {
        const float b = rewardConfig_.matchWinBonus;
        if (teamMode_ == TeamMode::Pairs) {
            const int32_t teamA = afterScores[0] + afterScores[2];
            const int32_t teamB = afterScores[1] + afterScores[3];
            if (teamA != teamB) {
                const float rA = (teamA > teamB) ? b : -b;
                outStepRewards[0] += rA;
                outStepRewards[2] += rA;
                outStepRewards[1] -= rA;
                outStepRewards[3] -= rA;
            }
        } else {
            constexpr std::array<float, 4> kPlaceWeights = {1.0f, 1.0f / 3.0f, -1.0f / 3.0f, -1.0f};
            for (int p = 0; p < kNumPlayers; ++p) {
                int strictlyBetter = 0;
                int equalCount = 0;
                for (int q = 0; q < kNumPlayers; ++q) {
                    if (afterScores[q] > afterScores[p]) {
                        ++strictlyBetter;
                    } else if (afterScores[q] == afterScores[p]) {
                        ++equalCount;
                    }
                }
                float avgWeight = 0.0f;
                for (int r = strictlyBetter; r < strictlyBetter + equalCount; ++r) {
                    avgWeight += kPlaceWeights[static_cast<size_t>(r)];
                }
                outStepRewards[p] += b * (avgWeight / static_cast<float>(equalCount));
            }
        }
    }
}

void JokerVectorEnv::GetObservations(
    std::span<float> outObsBatch,
    std::span<uint64_t> outLegalMasks,
    std::span<uint8_t> outCurrentPlayers,
    std::span<float> outOracleBatch,
    std::span<int16_t> outHistoryTokens,
    std::span<uint8_t> outHistoryLengths) const noexcept {
    const ptrdiff_t n = static_cast<ptrdiff_t>(slots_.size());
    constexpr size_t kObsStride = GameStateView::kTensorSize;
    constexpr size_t kSeqStride = DealHistoryBuffer::kMaxSeqLen * DealHistoryBuffer::kTokenDim;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(numThreads_) if (n >= 16)
#endif
    for (ptrdiff_t i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const EnvSlot& slot = slots_[idx];
        const PlayerId curP = slot.engine.GetCurrentPlayer();

        if (!outObsBatch.empty()) {
            slot.engine.WriteObservationTensor(curP, outObsBatch.data() + idx * kObsStride);
        }
        if (!outLegalMasks.empty()) {
            outLegalMasks[idx] = slot.engine.GetLegalActionsMask();
        }
        if (!outCurrentPlayers.empty()) {
            outCurrentPlayers[idx] = curP;
        }
        if (!outOracleBatch.empty()) {
            WriteOracleTensor(slot.engine.GetState(), curP, outOracleBatch.data() + idx * kOracleTensorSize);
        }
        if (!outHistoryTokens.empty()) {
            uint8_t* lenPtr = outHistoryLengths.empty() ? nullptr : &outHistoryLengths[idx];
            slot.history.WriteTokens(curP, outHistoryTokens.data() + idx * kSeqStride, lenPtr);
        } else if (!outHistoryLengths.empty()) {
            outHistoryLengths[idx] = slot.history.count;
        }
    }
}

void JokerVectorEnv::StepBatch(
    std::span<const uint8_t> actions,
    std::span<float> outNextObsBatch,
    std::span<uint64_t> outNextMasks,
    std::span<uint8_t> outNextPlayers,
    std::span<float> outRewards,
    std::span<uint8_t> outDealDones,
    std::span<uint8_t> outMatchDones,
    std::span<int32_t> outFinalScores,
    std::span<float> outNextOracleBatch,
    std::span<int16_t> outNextHistoryTokens,
    std::span<uint8_t> outNextHistoryLengths) noexcept {
    const ptrdiff_t n = static_cast<ptrdiff_t>(slots_.size());
    constexpr size_t kObsStride = GameStateView::kTensorSize;
    constexpr size_t kSeqStride = DealHistoryBuffer::kMaxSeqLen * DealHistoryBuffer::kTokenDim;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(numThreads_) if (n >= 16)
#endif
    for (ptrdiff_t i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        EnvSlot& slot = slots_[idx];
        JokerEngine& env = slot.engine;

        const GameState& stBefore = env.GetState();
        const auto prevScores = stBefore.totalScores;
        const int prevDeal = stBefore.dealNumber;
        const auto prevBids = stBefore.bids;
        const int prevTricksCompleted = stBefore.tricksCompletedInDeal;
        const bool wasLastTrickCard = (stBefore.phase == Phase::TrickPlay && stBefore.cardsInCurrentTrick == 3);

        ExecuteAndRecordStep(slot, actions[idx]);

        const GameState& stAfter = env.GetState();
        const auto afterScores = stAfter.totalScores;
        const bool matchDone = env.IsDone();
        const bool dealDone = matchDone || (stAfter.dealNumber != prevDeal);
        const bool trickCompleted = wasLastTrickCard && (dealDone || stAfter.tricksCompletedInDeal != prevTricksCompleted);

        if (!outRewards.empty()) {
            std::array<float, kNumPlayers> stepRewards{};
            ComputeStepRewards(
                prevScores,
                afterScores,
                trickCompleted && !dealDone,
                stAfter.lastTrickWinner,
                prevBids,
                stAfter.tricksTaken,
                dealDone,
                matchDone,
                stepRewards);

            float* rPtr = outRewards.data() + idx * kNumPlayers;
            for (int p = 0; p < kNumPlayers; ++p) {
                rPtr[p] = stepRewards[p];
            }
        }

        if (!outDealDones.empty()) {
            outDealDones[idx] = dealDone ? 1U : 0U;
        }
        if (!outMatchDones.empty()) {
            outMatchDones[idx] = matchDone ? 1U : 0U;
        }
        if (!outFinalScores.empty()) {
            int32_t* fPtr = outFinalScores.data() + idx * kNumPlayers;
            for (int p = 0; p < kNumPlayers; ++p) {
                fPtr[p] = afterScores[p];
            }
        }

        if (matchDone) {
            const uint64_t ep = ++slot.episodeCounter;
            uint64_t s256[4];
            GenerateEnvSeed256(idx, ep, s256);
            env.Reset256(preset_, s256, mode_, teamMode_, pulkaMode_);
            slot.history.Clear();
        } else if (dealDone) {
            slot.history.Clear();
        }

        const PlayerId nextP = env.GetCurrentPlayer();
        if (!outNextObsBatch.empty()) {
            env.WriteObservationTensor(nextP, outNextObsBatch.data() + idx * kObsStride);
        }
        if (!outNextMasks.empty()) {
            outNextMasks[idx] = env.GetLegalActionsMask();
        }
        if (!outNextPlayers.empty()) {
            outNextPlayers[idx] = nextP;
        }
        if (!outNextOracleBatch.empty()) {
            WriteOracleTensor(env.GetState(), nextP, outNextOracleBatch.data() + idx * kOracleTensorSize);
        }
        if (!outNextHistoryTokens.empty()) {
            uint8_t* lenPtr = outNextHistoryLengths.empty() ? nullptr : &outNextHistoryLengths[idx];
            slot.history.WriteTokens(nextP, outNextHistoryTokens.data() + idx * kSeqStride, lenPtr);
        } else if (!outNextHistoryLengths.empty()) {
            outNextHistoryLengths[idx] = slot.history.count;
        }
    }
}

ActionId JokerVectorEnv::SelectHeuristicAction(
    const JokerEngine& env,
    uint64_t& rngState) noexcept {
    // Заглушка: быстрый несмещённый случайный выбор среди легальных ходов
    ActionMask mask = env.GetLegalActionsMask();
    if (mask == 0ULL) return 0;

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

void JokerVectorEnv::AdvanceToHero(
    std::span<const uint8_t> heroSeats,
    std::span<float> outObsBatch,
    std::span<uint64_t> outLegalMasks,
    std::span<float> outOracleBatch,
    std::span<int16_t> outHistoryTokens,
    std::span<uint8_t> outHistoryLengths) noexcept {
    const ptrdiff_t n = static_cast<ptrdiff_t>(slots_.size());
    constexpr size_t kObsStride = GameStateView::kTensorSize;
    constexpr size_t kSeqStride = DealHistoryBuffer::kMaxSeqLen * DealHistoryBuffer::kTokenDim;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(numThreads_) if (n >= 16)
#endif
    for (ptrdiff_t i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        EnvSlot& slot = slots_[idx];
        JokerEngine& env = slot.engine;
        const PlayerId hero = static_cast<PlayerId>((!heroSeats.empty() ? heroSeats[idx] : 0U) % kNumPlayers);

        while (!env.IsDone() && env.GetCurrentPlayer() != hero) {
            const int prevDeal = env.GetState().dealNumber;
            const ActionId botAct = SelectHeuristicAction(env, slot.rngState);
            ExecuteAndRecordStep(slot, botAct);

            if (env.IsDone()) {
                const uint64_t ep = ++slot.episodeCounter;
                uint64_t s256[4];
                GenerateEnvSeed256(idx, ep, s256);
                env.Reset256(preset_, s256, mode_, teamMode_, pulkaMode_);
                slot.history.Clear();
            } else if (env.GetState().dealNumber != prevDeal) {
                slot.history.Clear();
            }
        }

        if (!outObsBatch.empty()) {
            env.WriteObservationTensor(hero, outObsBatch.data() + idx * kObsStride);
        }
        if (!outLegalMasks.empty()) {
            outLegalMasks[idx] = env.GetLegalActionsMask();
        }
        if (!outOracleBatch.empty()) {
            WriteOracleTensor(env.GetState(), hero, outOracleBatch.data() + idx * kOracleTensorSize);
        }
        if (!outHistoryTokens.empty()) {
            uint8_t* lenPtr = outHistoryLengths.empty() ? nullptr : &outHistoryLengths[idx];
            slot.history.WriteTokens(hero, outHistoryTokens.data() + idx * kSeqStride, lenPtr);
        } else if (!outHistoryLengths.empty()) {
            outHistoryLengths[idx] = slot.history.count;
        }
    }
}

void JokerVectorEnv::StepArena(
    std::span<const uint8_t> heroActions,
    std::span<const uint8_t> heroSeats,
    std::span<float> outNextObsBatch,
    std::span<uint64_t> outNextMasks,
    std::span<float> outHeroRewards,
    std::span<float> outAllRewards,
    std::span<uint8_t> outDealDones,
    std::span<uint8_t> outMatchDones,
    std::span<int32_t> outFinalScores,
    std::span<float> outNextOracleBatch,
    std::span<int16_t> outNextHistoryTokens,
    std::span<uint8_t> outNextHistoryLengths) noexcept {
    const ptrdiff_t n = static_cast<ptrdiff_t>(slots_.size());
    constexpr size_t kObsStride = GameStateView::kTensorSize;
    constexpr size_t kSeqStride = DealHistoryBuffer::kMaxSeqLen * DealHistoryBuffer::kTokenDim;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(numThreads_) if (n >= 16)
#endif
    for (ptrdiff_t i = 0; i < n; ++i) {
        const size_t idx = static_cast<size_t>(i);
        EnvSlot& slot = slots_[idx];
        JokerEngine& env = slot.engine;
        const PlayerId hero = static_cast<PlayerId>((!heroSeats.empty() ? heroSeats[idx] : 0U) % kNumPlayers);

        std::array<float, kNumPlayers> accumRewards{0.0f, 0.0f, 0.0f, 0.0f};
        bool anyDealDone = false;
        bool anyMatchDone = false;
        std::array<int32_t, kNumPlayers> finalMatchScores = env.GetState().totalScores;

        auto executeSingleStep = [&](ActionId act) noexcept {
            const GameState& stBefore = env.GetState();
            const auto prevScores = stBefore.totalScores;
            const int prevDeal = stBefore.dealNumber;
            const auto prevBids = stBefore.bids;
            const int prevTricksCompleted = stBefore.tricksCompletedInDeal;
            const bool wasLastTrickCard = (stBefore.phase == Phase::TrickPlay && stBefore.cardsInCurrentTrick == 3);

            ExecuteAndRecordStep(slot, act);

            const GameState& stAfter = env.GetState();
            const auto afterScores = stAfter.totalScores;
            const bool matchDone = env.IsDone();
            const bool dealDone = matchDone || (stAfter.dealNumber != prevDeal);
            const bool trickCompleted = wasLastTrickCard && (dealDone || stAfter.tricksCompletedInDeal != prevTricksCompleted);

            std::array<float, kNumPlayers> stepRewards{};
            ComputeStepRewards(
                prevScores,
                afterScores,
                trickCompleted && !dealDone,
                stAfter.lastTrickWinner,
                prevBids,
                stAfter.tricksTaken,
                dealDone,
                matchDone,
                stepRewards);

            for (int p = 0; p < kNumPlayers; ++p) {
                accumRewards[p] += stepRewards[p];
            }

            if (matchDone) {
                anyDealDone = true;
                anyMatchDone = true;
                finalMatchScores = afterScores;

                const uint64_t ep = ++slot.episodeCounter;
                uint64_t s256[4];
                GenerateEnvSeed256(idx, ep, s256);
                env.Reset256(preset_, s256, mode_, teamMode_, pulkaMode_);
                slot.history.Clear();
            } else if (dealDone) {
                anyDealDone = true;
                slot.history.Clear();
                if (!anyMatchDone) {
                    finalMatchScores = afterScores;
                }
            } else if (!anyMatchDone) {
                finalMatchScores = afterScores;
            }
        };

        executeSingleStep(heroActions[idx]);

        while (!env.IsDone() && env.GetCurrentPlayer() != hero) {
            const ActionId botAct = SelectHeuristicAction(env, slot.rngState);
            executeSingleStep(botAct);
        }

        if (!outHeroRewards.empty()) {
            outHeroRewards[idx] = accumRewards[hero];
        }
        if (!outAllRewards.empty()) {
            float* rPtr = outAllRewards.data() + idx * kNumPlayers;
            for (int p = 0; p < kNumPlayers; ++p) {
                rPtr[p] = accumRewards[p];
            }
        }
        if (!outDealDones.empty()) {
            outDealDones[idx] = anyDealDone ? 1U : 0U;
        }
        if (!outMatchDones.empty()) {
            outMatchDones[idx] = anyMatchDone ? 1U : 0U;
        }
        if (!outFinalScores.empty()) {
            int32_t* fPtr = outFinalScores.data() + idx * kNumPlayers;
            for (int p = 0; p < kNumPlayers; ++p) {
                fPtr[p] = finalMatchScores[p];
            }
        }
        if (!outNextObsBatch.empty()) {
            env.WriteObservationTensor(hero, outNextObsBatch.data() + idx * kObsStride);
        }
        if (!outNextMasks.empty()) {
            outNextMasks[idx] = env.GetLegalActionsMask();
        }
        if (!outNextOracleBatch.empty()) {
            WriteOracleTensor(env.GetState(), hero, outNextOracleBatch.data() + idx * kOracleTensorSize);
        }
        if (!outNextHistoryTokens.empty()) {
            uint8_t* lenPtr = outNextHistoryLengths.empty() ? nullptr : &outNextHistoryLengths[idx];
            slot.history.WriteTokens(hero, outNextHistoryTokens.data() + idx * kSeqStride, lenPtr);
        } else if (!outNextHistoryLengths.empty()) {
            outNextHistoryLengths[idx] = slot.history.count;
        }
    }
}

} // namespace joker::engine