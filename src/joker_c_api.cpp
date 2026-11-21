#include "joker_c_api.h"
#include "engine/DealHistory.hpp"
#include "engine/JokerEngine.hpp"
#include "engine/JokerVectorEnv.hpp"
#include <array>
#include <span>
#include <utility>

struct UndoEntry {
    joker::engine::JokerEngine engine{};
    joker::engine::DealHistoryBuffer history{};
};

struct JokerEnvHandle {
    static constexpr size_t kMaxUndo = 64;

    joker::engine::JokerEngine engine{};
    joker::engine::DealHistoryBuffer history{};
    std::array<UndoEntry, kMaxUndo> undoRing{};
    size_t undoHead{0};
    uint32_t undoCount{0};

    explicit JokerEnvHandle(const joker::engine::JokerEngine& eng) noexcept
        : engine(eng) {}

    void ClearHistoryAndUndo() noexcept {
        history.Clear();
        undoHead = 0;
        undoCount = 0;
    }

    void PushUndo() noexcept {
        undoRing[undoHead] = UndoEntry{engine, history};
        undoHead = (undoHead + 1U) % kMaxUndo;
        if (undoCount < kMaxUndo) {
            ++undoCount;
        }
    }

    bool PopUndo() noexcept {
        if (undoCount == 0U) {
            return false;
        }
        undoHead = (undoHead + kMaxUndo - 1U) % kMaxUndo;
        --undoCount;
        engine = undoRing[undoHead].engine;
        history = undoRing[undoHead].history;
        return true;
    }
};

struct JokerVecEnvHandle {
    joker::engine::JokerVectorEnv vec_env;
};

namespace {

constexpr joker::core::RulePreset ParseRulePreset(uint8_t v) noexcept {
    return static_cast<joker::core::RulePreset>(v % 3U);
}

constexpr joker::core::MatchMode ParseMatchMode(uint8_t v) noexcept {
    return (v == 1U) ? joker::core::MatchMode::Short : joker::core::MatchMode::Long;
}

constexpr joker::core::TeamMode ParseTeamMode(uint8_t v) noexcept {
    return (v == 1U) ? joker::core::TeamMode::Pairs : joker::core::TeamMode::Solo;
}

constexpr joker::core::PulkaMode ParsePulkaMode(uint8_t v) noexcept {
    return (v == 1U) ? joker::core::PulkaMode::Soft : joker::core::PulkaMode::Strict;
}

constexpr JokerStepError ToCError(joker::core::StepError err) noexcept {
    return static_cast<JokerStepError>(std::to_underlying(err));
}

template <typename T>
constexpr std::span<T> MakeSpan(T* ptr, size_t count) noexcept {
    return ptr ? std::span<T>(ptr, count) : std::span<T>{};
}

inline JokerStepError ExecuteSingleEnvStepChecked(JokerEnvHandle* env, uint8_t action_id) noexcept {
    if (!env) {
        return JOKER_ERR_NULL_HANDLE;
    }
    const joker::core::StepError err = env->engine.ValidateAction(action_id);
    if (err != joker::core::StepError::Ok) {
        return ToCError(err);
    }

    env->PushUndo();

    const joker::core::GameState& stBefore = env->engine.GetState();
    const int prevDeal = stBefore.dealNumber;
    const auto ev = joker::engine::DealHistoryBuffer::MakeEvent(stBefore, action_id);

    env->engine.Step(action_id);

    const bool dealChanged = env->engine.IsDone() || (env->engine.GetState().dealNumber != prevDeal);
    if (dealChanged) {
        env->history.Clear();
    } else if (ev.eventType != 0U) {
        env->history.Push(ev);
    }

    return JOKER_ERR_OK;
}

} // namespace

extern "C" {

JokerEnvHandle* joker_create(
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t seed) {
    return new JokerEnvHandle(
        joker::engine::JokerEngine(
            ParseRulePreset(rules_preset),
            seed,
            ParseMatchMode(match_mode),
            ParseTeamMode(team_mode),
            ParsePulkaMode(pulka_mode)));
}

void joker_destroy(JokerEnvHandle* env) {
    delete env;
}

JokerEnvHandle* joker_clone(const JokerEnvHandle* env) {
    if (!env) return nullptr;
    auto* cloned = new JokerEnvHandle(env->engine);
    cloned->history = env->history;
    return cloned;
}

JokerEnvHandle* joker_clone_determinized(
    const JokerEnvHandle* env,
    uint8_t observer_player,
    uint64_t rng_seed) {
    if (!env) return nullptr;
    auto* cloned = new JokerEnvHandle(env->engine);
    cloned->history = env->history;
    cloned->engine.Determinize(observer_player % 4U, rng_seed);
    return cloned;
}

void joker_determinize_inplace(
    JokerEnvHandle* env,
    uint8_t observer_player,
    uint64_t rng_seed) {
    if (!env) return;
    env->engine.Determinize(observer_player % 4U, rng_seed);
}

void joker_reset(
    JokerEnvHandle* env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t seed) {
    if (!env) return;
    env->engine.Reset(
        ParseRulePreset(rules_preset),
        seed,
        ParseMatchMode(match_mode),
        ParseTeamMode(team_mode),
        ParsePulkaMode(pulka_mode));
    env->ClearHistoryAndUndo();
}

void joker_reset256(
    JokerEnvHandle* env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    const uint64_t seed256[4]) {
    if (!env || !seed256) return;
    env->engine.Reset256(
        ParseRulePreset(rules_preset),
        std::span<const uint64_t, 4>(seed256, 4),
        ParseMatchMode(match_mode),
        ParseTeamMode(team_mode),
        ParsePulkaMode(pulka_mode));
    env->ClearHistoryAndUndo();
}

void joker_force_deck_order(JokerEnvHandle* env, const uint8_t cards38[38]) {
    if (!env || !cards38) return;
    env->engine.ForceDeckOrder(std::span<const uint8_t, joker::core::kMaxDeckSize>(cards38, 38));
}

void joker_begin_observed_match(
    JokerEnvHandle* env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint8_t dealer,
    uint8_t hero_seat) {
    if (!env) return;
    env->engine.BeginObservedMatch(
        ParseRulePreset(rules_preset),
        ParseMatchMode(match_mode),
        ParseTeamMode(team_mode),
        ParsePulkaMode(pulka_mode),
        dealer % 4U,
        hero_seat % 4U);
    env->ClearHistoryAndUndo();
}

int joker_set_observed_match_state(
    JokerEnvHandle* env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    int32_t deal_number,
    uint8_t dealer,
    uint8_t hero_seat,
    const int32_t total_scores4[4],
    const uint8_t pulka_clean4[4],
    const uint8_t pulka_all_passes4[4],
    const int16_t pulka_max_score_excl_last4[4],
    const int16_t pulka_max_non_pass_score_excl_last4[4]) {
    if (!env || !total_scores4 || !pulka_clean4 || !pulka_all_passes4 ||
        !pulka_max_score_excl_last4 || !pulka_max_non_pass_score_excl_last4) {
        return 0;
    }

    const bool ok = env->engine.SetObservedMatchState(
        ParseRulePreset(rules_preset),
        ParseMatchMode(match_mode),
        ParseTeamMode(team_mode),
        ParsePulkaMode(pulka_mode),
        deal_number,
        dealer % 4U,
        hero_seat % 4U,
        std::span<const int32_t, 4>(total_scores4, 4),
        std::span<const uint8_t, 4>(pulka_clean4, 4),
        std::span<const uint8_t, 4>(pulka_all_passes4, 4),
        std::span<const int16_t, 4>(pulka_max_score_excl_last4, 4),
        std::span<const int16_t, 4>(pulka_max_non_pass_score_excl_last4, 4));

    if (!ok) {
        return 0;
    }

    env->ClearHistoryAndUndo();
    return 1;
}

int joker_set_observed_deal(
    JokerEnvHandle* env,
    uint64_t hero_hand_mask,
    uint8_t upcard_or_invalid,
    uint8_t trump_suit_or_none) {
    if (!env) return 0;
    const UndoEntry backup{env->engine, env->history};
    if (!env->engine.SetObservedDeal(hero_hand_mask, upcard_or_invalid, trump_suit_or_none)) {
        return 0;
    }
    env->undoRing[env->undoHead] = backup;
    env->undoHead = (env->undoHead + 1U) % JokerEnvHandle::kMaxUndo;
    if (env->undoCount < JokerEnvHandle::kMaxUndo) {
        ++env->undoCount;
    }
    env->history.Clear();
    return 1;
}

int joker_set_observed_remaining_six(
    JokerEnvHandle* env,
    uint64_t full_nine_hand_mask) {
    if (!env) return 0;
    const UndoEntry backup{env->engine, env->history};
    if (!env->engine.SetObservedRemainingSix(full_nine_hand_mask)) {
        return 0;
    }
    env->undoRing[env->undoHead] = backup;
    env->undoHead = (env->undoHead + 1U) % JokerEnvHandle::kMaxUndo;
    if (env->undoCount < JokerEnvHandle::kMaxUndo) {
        ++env->undoCount;
    }
    return 1;
}

int joker_is_done(const JokerEnvHandle* env) {
    return (!env || env->engine.IsDone()) ? 1 : 0;
}

uint8_t joker_get_phase(const JokerEnvHandle* env) {
    if (!env) return std::to_underlying(joker::core::Phase::MatchFinished);
    return std::to_underlying(env->engine.GetPhase());
}

uint8_t joker_get_current_player(const JokerEnvHandle* env) {
    return env ? env->engine.GetCurrentPlayer() : 0U;
}

uint64_t joker_get_legal_actions_mask(const JokerEnvHandle* env) {
    return env ? env->engine.GetLegalActionsMask() : 0ULL;
}

int joker_step(JokerEnvHandle* env, uint8_t action_id) {
    return (ExecuteSingleEnvStepChecked(env, action_id) == JOKER_ERR_OK) ? 1 : 0;
}

JokerStepError joker_validate_action(const JokerEnvHandle* env, uint8_t action_id) {
    if (!env) return JOKER_ERR_NULL_HANDLE;
    return ToCError(env->engine.ValidateAction(action_id));
}

JokerStepError joker_step_checked(JokerEnvHandle* env, uint8_t action_id) {
    return ExecuteSingleEnvStepChecked(env, action_id);
}

int joker_undo(JokerEnvHandle* env) {
    return (env && env->PopUndo()) ? 1 : 0;
}

uint32_t joker_can_undo_steps(const JokerEnvHandle* env) {
    return env ? env->undoCount : 0U;
}

void joker_write_history_tokens(
    const JokerEnvHandle* env,
    uint8_t observer_player,
    int16_t* out_tokens,
    uint8_t* out_length) {
    if (!env || !out_tokens) return;
    env->history.WriteTokens(observer_player % 4U, out_tokens, out_length);
}

uint8_t joker_select_heuristic_action(const JokerEnvHandle* env, uint64_t rng_seed) {
    if (!env) return 0U;
    uint64_t rng = (rng_seed == 0ULL) ? 0x9E3779B97F4A7C15ULL : rng_seed;
    return joker::engine::JokerVectorEnv::SelectHeuristicAction(env->engine, rng);
}

size_t joker_tensor_size(void) {
    return joker::core::GameStateView::kTensorSize;
}

size_t joker_oracle_tensor_size(void) {
    return joker::engine::JokerVectorEnv::kOracleTensorSize;
}

size_t joker_history_seq_len(void) {
    return joker::engine::DealHistoryBuffer::kMaxSeqLen;
}

size_t joker_history_token_dim(void) {
    return joker::engine::DealHistoryBuffer::kTokenDim;
}

void joker_write_observation_tensor(
    const JokerEnvHandle* env,
    uint8_t observer_player,
    float* out_buffer) {
    if (!env || !out_buffer) return;
    env->engine.WriteObservationTensor(observer_player % 4U, out_buffer);
}

void joker_get_total_scores(const JokerEnvHandle* env, int32_t* out_scores4) {
    if (!env || !out_scores4) return;
    const auto& scores = env->engine.GetState().totalScores;
    for (int i = 0; i < 4; ++i) {
        out_scores4[i] = scores[i];
    }
}

void joker_get_team_scores(const JokerEnvHandle* env, int32_t* out_team_scores2) {
    if (!env || !out_team_scores2) return;
    const auto teams = env->engine.GetTeamScores();
    out_team_scores2[0] = teams[0];
    out_team_scores2[1] = teams[1];
}

int32_t joker_get_deal_number(const JokerEnvHandle* env) {
    return env ? env->engine.GetState().dealNumber : 0;
}

void joker_get_snapshot(const JokerEnvHandle* env, JokerSnapshot* s) {
    if (!env || !s) return;
    const auto& st = env->engine.GetState();
    s->played_cards = st.playedCardsHistory;
    s->deal_number = st.dealNumber;
    s->cards_in_deal = st.cardsInDeal;
    s->rules = std::to_underlying(st.rules);
    s->match_mode = std::to_underlying(st.mode);
    s->team_mode = std::to_underlying(st.teamMode);
    s->pulka_mode = std::to_underlying(st.pulkaMode);
    s->phase = std::to_underlying(st.phase);
    s->dealer = st.dealer;
    s->current_player = st.currentPlayer;
    s->trick_leader = st.trickLeader;
    s->last_trick_winner = st.lastTrickWinner;
    s->trump_suit = std::to_underlying(st.trumpSuit);
    s->upcard = st.upcard;
    s->cards_in_trick = st.cardsInCurrentTrick;
    s->observed_flags = st.observedFlags;
    s->hero_seat = st.heroSeat;

    for (int i = 0; i < 4; ++i) {
        s->hands[i] = st.hands[i];
        s->bids[i] = st.bids[i];
        s->tricks_taken[i] = st.tricksTaken[i];
        s->total_scores[i] = st.totalScores[i];
        s->pulka_clean[i] = st.pulkaClean[i] ? 1 : 0;
        s->trick_players[i] = st.currentTrick[i].player;
        s->trick_cards[i] = st.currentTrick[i].play.cardId;
        s->trick_joker_actions[i] = std::to_underlying(st.currentTrick[i].play.jokerAction);
        s->trick_req_suits[i] = std::to_underlying(st.currentTrick[i].play.requestedSuit);
    }
}

// ============================================================================
// Vectorized Environment C-API
// ============================================================================

JokerVecEnvHandle* joker_vec_create(
    size_t num_envs,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t base_seed,
    int num_threads) {
    if (num_envs == 0) num_envs = 1;
    return new JokerVecEnvHandle{
        joker::engine::JokerVectorEnv(
            num_envs,
            ParseRulePreset(rules_preset),
            base_seed,
            num_threads,
            ParseMatchMode(match_mode),
            ParseTeamMode(team_mode),
            ParsePulkaMode(pulka_mode))
    };
}

void joker_vec_destroy(JokerVecEnvHandle* vec_env) {
    delete vec_env;
}

size_t joker_vec_size(const JokerVecEnvHandle* vec_env) {
    return vec_env ? vec_env->vec_env.Size() : 0;
}

void joker_vec_set_reward_config(JokerVecEnvHandle* vec_env, const JokerRewardConfig* config) {
    if (!vec_env || !config) return;
    vec_env->vec_env.SetRewardConfig(joker::engine::RewardConfig{
        .scoreScale = config->score_scale,
        .relativeWeight = config->relative_weight,
        .matchWinBonus = config->match_win_bonus,
        .dealExactBonus = config->deal_exact_bonus,
        .trickWinShaping = config->trick_win_shaping
    });
}

void joker_vec_set_auto_reset(JokerVecEnvHandle* vec_env, int enabled) {
    if (!vec_env) return;
    vec_env->vec_env.SetAutoReset(enabled != 0);
}

void joker_vec_set_stats_enabled(JokerVecEnvHandle* vec_env, int enabled) {
    if (!vec_env) return;
    vec_env->vec_env.SetStatsEnabled(enabled != 0);
}

void joker_vec_reset_stats(JokerVecEnvHandle* vec_env) {
    if (!vec_env) return;
    vec_env->vec_env.ResetStats();
}

void joker_vec_get_stats(const JokerVecEnvHandle* vec_env, JokerPlayerStats out_stats4[4]) {
    if (!vec_env || !out_stats4) return;
    vec_env->vec_env.GetAggregatedStats(std::span<JokerPlayerStats, 4>(out_stats4, 4));
}

void joker_vec_reset(
    JokerVecEnvHandle* vec_env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t base_seed) {
    if (!vec_env) return;
    vec_env->vec_env.Reset(
        ParseRulePreset(rules_preset),
        base_seed,
        ParseMatchMode(match_mode),
        ParseTeamMode(team_mode),
        ParsePulkaMode(pulka_mode));
}

void joker_vec_reset_with_seeds(
    JokerVecEnvHandle* vec_env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    const uint64_t* seeds_per_env) {
    if (!vec_env || !seeds_per_env) return;
    const size_t n = vec_env->vec_env.Size();
    vec_env->vec_env.ResetWithSeeds(
        ParseRulePreset(rules_preset),
        ParseMatchMode(match_mode),
        ParseTeamMode(team_mode),
        ParsePulkaMode(pulka_mode),
        std::span<const uint64_t>(seeds_per_env, n));
}

void joker_vec_fork_from_env(
    JokerVecEnvHandle* vec_env,
    const JokerEnvHandle* root_env,
    uint8_t observer_player,
    uint64_t base_seed,
    float* out_obs_batch,
    uint64_t* out_legal_masks,
    float* out_oracle_batch) {
    if (!vec_env || !root_env) return;
    const size_t n = vec_env->vec_env.Size();
    constexpr size_t kObsDim = joker::core::GameStateView::kTensorSize;
    constexpr size_t kOracleDim = joker::engine::JokerVectorEnv::kOracleTensorSize;

    vec_env->vec_env.ForkAndDeterminize(
        root_env->engine,
        observer_player % 4U,
        base_seed,
        MakeSpan(out_obs_batch, n * kObsDim),
        MakeSpan(out_legal_masks, n),
        MakeSpan(out_oracle_batch, n * kOracleDim),
        &root_env->history);
}

void joker_vec_get_observations(
    const JokerVecEnvHandle* vec_env,
    float* out_obs_batch,
    uint64_t* out_legal_masks,
    uint8_t* out_current_players,
    float* out_oracle_batch,
    int16_t* out_history_tokens,
    uint8_t* out_history_lengths) {
    if (!vec_env) return;
    const size_t n = vec_env->vec_env.Size();
    constexpr size_t kObsDim = joker::core::GameStateView::kTensorSize;
    constexpr size_t kOracleDim = joker::engine::JokerVectorEnv::kOracleTensorSize;
    constexpr size_t kSeqStride = joker::engine::DealHistoryBuffer::kMaxSeqLen *
                                  joker::engine::DealHistoryBuffer::kTokenDim;

    vec_env->vec_env.GetObservations(
        MakeSpan(out_obs_batch, n * kObsDim),
        MakeSpan(out_legal_masks, n),
        MakeSpan(out_current_players, n),
        MakeSpan(out_oracle_batch, n * kOracleDim),
        MakeSpan(out_history_tokens, n * kSeqStride),
        MakeSpan(out_history_lengths, n));
}

void joker_vec_step(
    JokerVecEnvHandle* vec_env,
    const uint8_t* actions,
    float* out_next_obs_batch,
    uint64_t* out_next_masks,
    uint8_t* out_next_players,
    float* out_rewards,
    uint8_t* out_deal_dones,
    uint8_t* out_match_dones,
    int32_t* out_final_scores,
    float* out_next_oracle_batch,
    int16_t* out_next_history_tokens,
    uint8_t* out_next_history_lengths) {
    if (!vec_env || !actions) return;
    const size_t n = vec_env->vec_env.Size();
    constexpr size_t kObsDim = joker::core::GameStateView::kTensorSize;
    constexpr size_t kOracleDim = joker::engine::JokerVectorEnv::kOracleTensorSize;
    constexpr size_t kSeqStride = joker::engine::DealHistoryBuffer::kMaxSeqLen *
                                  joker::engine::DealHistoryBuffer::kTokenDim;

    vec_env->vec_env.StepBatch(
        std::span<const uint8_t>(actions, n),
        MakeSpan(out_next_obs_batch, n * kObsDim),
        MakeSpan(out_next_masks, n),
        MakeSpan(out_next_players, n),
        MakeSpan(out_rewards, n * 4U),
        MakeSpan(out_deal_dones, n),
        MakeSpan(out_match_dones, n),
        MakeSpan(out_final_scores, n * 4U),
        MakeSpan(out_next_oracle_batch, n * kOracleDim),
        MakeSpan(out_next_history_tokens, n * kSeqStride),
        MakeSpan(out_next_history_lengths, n));
}

void joker_vec_advance_to_hero(
    JokerVecEnvHandle* vec_env,
    const uint8_t* hero_seats,
    float* out_obs_batch,
    uint64_t* out_legal_masks,
    float* out_oracle_batch,
    int16_t* out_history_tokens,
    uint8_t* out_history_lengths) {
    if (!vec_env) return;
    const size_t n = vec_env->vec_env.Size();
    constexpr size_t kObsDim = joker::core::GameStateView::kTensorSize;
    constexpr size_t kOracleDim = joker::engine::JokerVectorEnv::kOracleTensorSize;
    constexpr size_t kSeqStride = joker::engine::DealHistoryBuffer::kMaxSeqLen *
                                  joker::engine::DealHistoryBuffer::kTokenDim;

    vec_env->vec_env.AdvanceToHero(
        MakeSpan(hero_seats, n),
        MakeSpan(out_obs_batch, n * kObsDim),
        MakeSpan(out_legal_masks, n),
        MakeSpan(out_oracle_batch, n * kOracleDim),
        MakeSpan(out_history_tokens, n * kSeqStride),
        MakeSpan(out_history_lengths, n));
}

void joker_vec_step_arena(
    JokerVecEnvHandle* vec_env,
    const uint8_t* hero_actions,
    const uint8_t* hero_seats,
    float* out_next_obs_batch,
    uint64_t* out_next_masks,
    float* out_hero_rewards,
    float* out_all_rewards,
    uint8_t* out_deal_dones,
    uint8_t* out_match_dones,
    int32_t* out_final_scores,
    float* out_next_oracle_batch,
    int16_t* out_next_history_tokens,
    uint8_t* out_next_history_lengths) {
    if (!vec_env || !hero_actions) return;
    const size_t n = vec_env->vec_env.Size();
    constexpr size_t kObsDim = joker::core::GameStateView::kTensorSize;
    constexpr size_t kOracleDim = joker::engine::JokerVectorEnv::kOracleTensorSize;
    constexpr size_t kSeqStride = joker::engine::DealHistoryBuffer::kMaxSeqLen *
                                  joker::engine::DealHistoryBuffer::kTokenDim;

    vec_env->vec_env.StepArena(
        std::span<const uint8_t>(hero_actions, n),
        MakeSpan(hero_seats, n),
        MakeSpan(out_next_obs_batch, n * kObsDim),
        MakeSpan(out_next_masks, n),
        MakeSpan(out_hero_rewards, n),
        MakeSpan(out_all_rewards, n * 4U),
        MakeSpan(out_deal_dones, n),
        MakeSpan(out_match_dones, n),
        MakeSpan(out_final_scores, n * 4U),
        MakeSpan(out_next_oracle_batch, n * kOracleDim),
        MakeSpan(out_next_history_tokens, n * kSeqStride),
        MakeSpan(out_next_history_lengths, n));
}

} // extern "C"