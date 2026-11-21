#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__EMSCRIPTEN__)
#  include <emscripten/emscripten.h>
#  define JOKER_API EMSCRIPTEN_KEEPALIVE
#elif defined(_WIN32)
#  if defined(JOKER_CORE_STATIC)
#    define JOKER_API
#  elif defined(JOKER_CORE_EXPORTS)
#    define JOKER_API __declspec(dllexport)
#  else
#    define JOKER_API __declspec(dllimport)
#  endif
#else
#  define JOKER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct JokerEnvHandle JokerEnvHandle;
typedef struct JokerVecEnvHandle JokerVecEnvHandle;

// ============================================================================
// Каноническая нумерация ActionId (всего 46 действий: 0..45)
//   Phase::ChoosingTrump (0): 0=♠, 1=♣, 2=♦, 3=♥, 4=Б/К (None)
//   Phase::Bidding       (1): 0..cards_in_deal (количество взяток; висяк дилера снят)
//   Phase::TrickPlay     (2):
//     0..35  : обычная карта (CardId = suit*9 + rank; 6♠=0 и 6♣=9 не в 36-карточной колоде)
//     36     : Джокер PlayWin (ответ во взятку: побить)
//     37     : Джокер PlayLow (ответ во взятку: сбросить по дну)
//     38..41 : Джокер LeadHighSuit (первый ход по высшей масти: 38=♠, 39=♣, 40=♦, 41=♥)
//     42..45 : Джокер LeadLow      (первый ход по меньшей масти: 42=♠, 43=♣, 44=♦, 45=♥)
// ============================================================================

typedef enum {
    JOKER_ERR_OK = 0,
    JOKER_ERR_NULL_HANDLE = 1,
    JOKER_ERR_MATCH_FINISHED = 2,
    JOKER_ERR_WAITING_FOR_DEAL_INPUT = 3,
    JOKER_ERR_INVALID_ACTION_ID = 4,
    JOKER_ERR_FORBIDDEN_DEALER_BID = 5,
    JOKER_ERR_CARD_NOT_IN_HAND = 6,
    JOKER_ERR_CARD_ALREADY_PLAYED = 7,
    JOKER_ERR_CARD_IN_OTHER_KNOWN_HAND = 8,
    JOKER_ERR_CARD_IS_UPCARD = 9,
    JOKER_ERR_MUST_FOLLOW_SUIT = 10,
    JOKER_ERR_MUST_PLAY_TRUMP = 11,
    JOKER_ERR_MUST_PLAY_HIGHEST = 12,
    JOKER_ERR_VIOLATES_KNOWN_VOID_OR_BOUND = 13,
    JOKER_ERR_NO_JOKER_AVAILABLE = 14
} JokerStepError;

// ============================================================================
// 1. Одиночная среда (Single-Env API + Observed Mode + Undo + ISMCTS / PIMC)
// ============================================================================

// rules_preset: 0 = Popular, 1 = Minus, 2 = Classic
// match_mode:   0 = Long (24 сдачи), 1 = Short (8 сдач по 9 карт)
// team_mode:    0 = Solo (каждый за себя), 1 = Pairs (0+2 против 1+3)
// pulka_mode:   0 = Strict (строгий расчёт пульки), 1 = Soft (мягкий: сосед над вами over=(P+3)%4)
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

// Режим живого стола (Observed / Live Table Mode)
JOKER_API void joker_begin_observed_match(
    JokerEnvHandle* env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint8_t dealer,
    uint8_t hero_seat
);
// Возвращает 1 при успехе, 0 при невалидных входных данных.
// upcard_or_invalid: 0..37 или 255 (нет вскрытой карты / девятки с гаданием).
// trump_suit_or_none: 0..4 (или 255 для автовывода из upcard).
JOKER_API int joker_set_observed_deal(
    JokerEnvHandle* env,
    uint64_t hero_hand_mask,
    uint8_t upcard_or_invalid,
    uint8_t trump_suit_or_none
);
// Возвращает 1 при успехе, 0 при невалидных данных (проверяет подмножество прежних 3 карт и размер 9).
JOKER_API int joker_set_observed_remaining_six(
    JokerEnvHandle* env,
    uint64_t full_nine_hand_mask
);

JOKER_API int joker_is_done(const JokerEnvHandle* env);
JOKER_API uint8_t joker_get_phase(const JokerEnvHandle* env);
JOKER_API uint8_t joker_get_current_player(const JokerEnvHandle* env);
JOKER_API uint64_t joker_get_legal_actions_mask(const JokerEnvHandle* env);

JOKER_API int joker_step(JokerEnvHandle* env, uint8_t action_id);
JOKER_API JokerStepError joker_validate_action(const JokerEnvHandle* env, uint8_t action_id);
JOKER_API JokerStepError joker_step_checked(JokerEnvHandle* env, uint8_t action_id);

// Откат хода (кольцевой буфер на 64 снимка) и история сдачи для Transformer/LSTM
JOKER_API int joker_undo(JokerEnvHandle* env);
JOKER_API uint32_t joker_can_undo_steps(const JokerEnvHandle* env);
JOKER_API void joker_write_history_tokens(
    const JokerEnvHandle* env,
    uint8_t observer_player,
    int16_t* out_tokens,
    uint8_t* out_length
);

// Встроенная быстрая C++ эвристика
JOKER_API uint8_t joker_select_heuristic_action(const JokerEnvHandle* env, uint64_t rng_seed);

JOKER_API size_t joker_tensor_size(void);        // 320 floats
JOKER_API size_t joker_oracle_tensor_size(void); // 160 floats
JOKER_API size_t joker_history_seq_len(void);    // 48 шагов в сдаче
JOKER_API size_t joker_history_token_dim(void);  // 8 int16 признаков на шаг

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
    int32_t  deal_number;
    int32_t  cards_in_deal;
    int32_t  bids[4];
    int32_t  tricks_taken[4];
    int32_t  total_scores[4];
    uint8_t  rules;
    uint8_t  match_mode;
    uint8_t  team_mode;
    uint8_t  pulka_mode;
    uint8_t  phase;
    uint8_t  dealer;
    uint8_t  current_player;
    uint8_t  trick_leader;
    uint8_t  last_trick_winner;
    uint8_t  trump_suit;
    uint8_t  upcard;
    uint8_t  cards_in_trick;
    uint8_t  pulka_clean[4];
    uint8_t  trick_players[4];
    uint8_t  trick_cards[4];
    uint8_t  trick_joker_actions[4];
    uint8_t  trick_req_suits[4];
} JokerSnapshot;

JOKER_API void joker_get_snapshot(const JokerEnvHandle* env, JokerSnapshot* out_snap);

// ============================================================================
// 2. Векторизованная среда (RL / Transformers / LSTM / Hybrid MCTS / Arena)
// ============================================================================

typedef struct {
    float score_scale;
    float relative_weight;
    float match_win_bonus;
    float deal_exact_bonus;
    float trick_win_shaping;
} JokerRewardConfig;

// Статистика игрока за местом 0..3 для Арены.
// Правило распределения мест при ничьей:
//   - Solo: competition ranking (place_counts[strictly_better]++);
//           place_points += (strictly_better == 0) ? (1.0 / tied_for_first) : 0.0.
//   - Pairs: победившая пара получает place_counts[0]++ и place_points += 1.0,
//           проигравшая — place_counts[1]++; при ничьей все 4 получают place_counts[0]++ и 0.5.
typedef struct {
    uint64_t matches_played;
    uint64_t place_counts[4];
    double   place_points;
    int64_t  total_match_score_sum;
    double   total_match_score_sq_sum;
    int32_t  max_match_score;
    int32_t  min_match_score;

    uint64_t deals_played;
    int32_t  max_deal_score;
    int32_t  min_deal_score;
    uint64_t exact_bids;
    uint64_t overbids;
    uint64_t underbids;
    uint64_t shangas;
    uint64_t large_bids_count;
    uint64_t large_bids_exact;

    uint64_t deals_1_8_count;
    uint64_t bids_sum_1_8;
    uint64_t tricks_sum_1_8;
    uint64_t deals_9_count;
    uint64_t bids_sum_9;
    uint64_t tricks_sum_9;

    uint64_t nines_trump_choices;
    uint64_t nines_no_trump_choices;

    uint64_t joker_lead_high;
    uint64_t joker_lead_low;
    uint64_t joker_play_win;
    uint64_t joker_play_low;

    uint64_t pulkas_played;
    uint64_t pulkas_clean;
    uint64_t pulkas_bonus_received;
    uint64_t pulkas_penalty_received;
    uint64_t matches_multi_bonus;
} JokerPlayerStats;

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
JOKER_API void joker_vec_set_auto_reset(JokerVecEnvHandle* vec_env, int enabled);

// Сбор статистики Арены (по умолчанию выключен для нулевого оверхеда в RL)
JOKER_API void joker_vec_set_stats_enabled(JokerVecEnvHandle* vec_env, int enabled);
JOKER_API void joker_vec_reset_stats(JokerVecEnvHandle* vec_env);
JOKER_API void joker_vec_get_stats(const JokerVecEnvHandle* vec_env, JokerPlayerStats out_stats4[4]);

JOKER_API void joker_vec_reset(
    JokerVecEnvHandle* vec_env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    uint64_t base_seed
);

// Зеркальные турниры (Duplicate Matches): явный массив из num_envs сидов
JOKER_API void joker_vec_reset_with_seeds(
    JokerVecEnvHandle* vec_env,
    uint8_t rules_preset,
    uint8_t match_mode,
    uint8_t team_mode,
    uint8_t pulka_mode,
    const uint64_t* seeds_per_env
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