#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
  #define JOKER_API __declspec(dllexport)
#else
  #define JOKER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct JokerEnvHandle JokerEnvHandle;
typedef struct JokerVecEnvHandle JokerVecEnvHandle;

// ============================================================================
// 1. Одиночная среда (Single-Env API + гибридный ISMCTS / PIMC поиск)
// ============================================================================

// rules_preset: 0 = Popular, 1 = Minus, 2 = Classic
// match_mode:   0 = Long (24 сдачи), 1 = Short (8 сдач по 9 карт)
// team_mode:    0 = Solo (каждый за себя), 1 = Pairs (0+2 против 1+3)
// pulka_mode:   0 = Strict (строгий расчёт пульки), 1 = Soft (мягкий расчёт пульки)
JOKER_API JokerEnvHandle* joker_create(
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t seed
);

JOKER_API void joker_destroy(JokerEnvHandle* env);
JOKER_API JokerEnvHandle* joker_clone(const JokerEnvHandle* env);

JOKER_API JokerEnvHandle* joker_clone_determinized(
    const JokerEnvHandle* env,
    uint8_t observer_player,
    uint64_t rng_seed
);
JOKER_API void joker_determinize_inplace(
    JokerEnvHandle* env,
    uint8_t observer_player,
    uint64_t rng_seed
);

JOKER_API void joker_reset(
    JokerEnvHandle* env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t seed
);
JOKER_API void joker_reset256(
    JokerEnvHandle* env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    const uint64_t seed256[4]
);
JOKER_API void joker_force_deck_order(JokerEnvHandle* env, const uint8_t cards38[38]);

JOKER_API int joker_is_done(const JokerEnvHandle* env);
JOKER_API uint8_t joker_get_phase(const JokerEnvHandle* env);
JOKER_API uint8_t joker_get_current_player(const JokerEnvHandle* env);
JOKER_API uint64_t joker_get_legal_actions_mask(const JokerEnvHandle* env);

JOKER_API int joker_step(JokerEnvHandle* env, uint8_t action_id);

JOKER_API size_t joker_tensor_size(void);          // 384 floats
JOKER_API size_t joker_oracle_tensor_size(void);   // 160 floats
JOKER_API size_t joker_history_seq_len(void);      // 48 шагов в сдаче
JOKER_API size_t joker_history_token_dim(void);    // 8 int16 признаков на шаг

JOKER_API void joker_write_observation_tensor(
    const JokerEnvHandle* env,
    uint8_t observer_player,
    float* out_buffer
);

JOKER_API void joker_get_total_scores(const JokerEnvHandle* env, int32_t* out_scores4);
JOKER_API void joker_get_team_scores(const JokerEnvHandle* env, int32_t* out_team_scores2);
JOKER_API int32_t joker_get_deal_number(const JokerEnvHandle* env);

typedef struct {
    uint64_t hands[4];
    uint64_t played_cards;
    int32_t deal_number;
    int32_t cards_in_deal;
    int32_t bids[4];
    int32_t tricks_taken[4];
    int32_t total_scores[4];
    uint8_t rules;
    uint8_t match_mode;
    uint8_t team_mode;
    uint8_t pulka_mode;
    uint8_t phase;
    uint8_t dealer;
    uint8_t current_player;
    uint8_t trick_leader;
    uint8_t last_trick_winner;
    uint8_t trump_suit;
    uint8_t upcard;
    uint8_t cards_in_trick;
    uint8_t pulka_clean[4];
    uint8_t trick_players[4];
    uint8_t trick_cards[4];
    uint8_t trick_joker_actions[4];
    uint8_t trick_req_suits[4];
} JokerSnapshot;

JOKER_API void joker_get_snapshot(const JokerEnvHandle* env, JokerSnapshot* out_snap);

// ============================================================================
// 2. Векторизованная среда (RL / Transformers / LSTM / Hybrid MCTS / OpenMP)
// ============================================================================

typedef struct {
    float score_scale;
    float relative_weight;
    float match_win_bonus;
    float deal_exact_bonus;
    float trick_win_shaping;
} JokerRewardConfig;

JOKER_API JokerVecEnvHandle* joker_vec_create(
    size_t num_envs,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t base_seed,
    int num_threads
);

JOKER_API void joker_vec_destroy(JokerVecEnvHandle* vec_env);
JOKER_API size_t joker_vec_size(const JokerVecEnvHandle* vec_env);
JOKER_API void joker_vec_set_reward_config(JokerVecEnvHandle* vec_env, const JokerRewardConfig* config);

JOKER_API void joker_vec_reset(
    JokerVecEnvHandle* vec_env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t base_seed
);

JOKER_API void joker_vec_fork_from_env(
    JokerVecEnvHandle* vec_env,
    const JokerEnvHandle* root_env,
    uint8_t observer_player,
    uint64_t base_seed,
    float* out_obs_batch,
    uint64_t* out_legal_masks,
    float* out_oracle_batch
);

JOKER_API void joker_vec_get_observations(
    const JokerVecEnvHandle* vec_env,
    float* out_obs_batch,
    uint64_t* out_legal_masks,
    uint8_t* out_current_players,
    float* out_oracle_batch,
    int16_t* out_history_tokens,
    uint8_t* out_history_lengths
);

JOKER_API void joker_vec_step(
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
    uint8_t* out_next_history_lengths
);

JOKER_API void joker_vec_advance_to_hero(
    JokerVecEnvHandle* vec_env,
    const uint8_t* hero_seats,
    float* out_obs_batch,
    uint64_t* out_legal_masks,
    float* out_oracle_batch,
    int16_t* out_history_tokens,
    uint8_t* out_history_lengths
);

JOKER_API void joker_vec_step_arena(
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
    uint8_t* out_next_history_lengths
);

#ifdef __cplusplus
}
#endif