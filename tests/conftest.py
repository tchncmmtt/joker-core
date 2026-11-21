import ctypes
import os
import platform
from pathlib import Path
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

# ============================================================================
# 1. Загрузка библиотеки и описание C ABI
# ============================================================================

def find_library() -> Path:
    env_override = os.environ.get("JOKER_LIB_PATH")
    if env_override and Path(env_override).exists():
        return Path(env_override)

    root = Path(__file__).resolve().parent.parent
    lib_name = "joker_core.dll" if platform.system() == "Windows" else "libjoker_core.so"
    if platform.system() == "Darwin":
        lib_name = "libjoker_core.dylib"

    candidates = [
        root / "build" / lib_name,
        root / "build" / "Release" / lib_name,
        root / "cmake-build-release" / lib_name,
        root / "cmake-build-debug" / lib_name,
        root / lib_name,
        ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Не найдена библиотека {lib_name}. Сначала выполните сборку:\n"
        "cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
    )


LIB_PATH = find_library()
lib = ctypes.CDLL(str(LIB_PATH))


class JokerSnapshot(ctypes.Structure):
    _fields_ = [
        ("hands", ctypes.c_uint64 * 4),
        ("played_cards", ctypes.c_uint64),
        ("deal_number", ctypes.c_int32),
        ("cards_in_deal", ctypes.c_int32),
        ("bids", ctypes.c_int32 * 4),
        ("tricks_taken", ctypes.c_int32 * 4),
        ("total_scores", ctypes.c_int32 * 4),
        ("rules", ctypes.c_uint8),
        ("match_mode", ctypes.c_uint8),
        ("team_mode", ctypes.c_uint8),
        ("pulka_mode", ctypes.c_uint8),
        ("phase", ctypes.c_uint8),
        ("dealer", ctypes.c_uint8),
        ("current_player", ctypes.c_uint8),
        ("trick_leader", ctypes.c_uint8),
        ("last_trick_winner", ctypes.c_uint8),
        ("trump_suit", ctypes.c_uint8),
        ("upcard", ctypes.c_uint8),
        ("cards_in_trick", ctypes.c_uint8),
        ("pulka_clean", ctypes.c_uint8 * 4),
        ("trick_players", ctypes.c_uint8 * 4),
        ("trick_cards", ctypes.c_uint8 * 4),
        ("trick_joker_actions", ctypes.c_uint8 * 4),
        ("trick_req_suits", ctypes.c_uint8 * 4),
    ]


class JokerRewardConfig(ctypes.Structure):
    _fields_ = [
        ("score_scale", ctypes.c_float),
        ("relative_weight", ctypes.c_float),
        ("match_win_bonus", ctypes.c_float),
        ("deal_exact_bonus", ctypes.c_float),
        ("trick_win_shaping", ctypes.c_float),
    ]


class JokerPlayerStats(ctypes.Structure):
    _fields_ = [
        ("matches_played", ctypes.c_uint64),
        ("place_counts", ctypes.c_uint64 * 4),
        ("place_points", ctypes.c_double),
        ("total_match_score_sum", ctypes.c_int64),
        ("total_match_score_sq_sum", ctypes.c_double),
        ("max_match_score", ctypes.c_int32),
        ("min_match_score", ctypes.c_int32),
        ("deals_played", ctypes.c_uint64),
        ("max_deal_score", ctypes.c_int32),
        ("min_deal_score", ctypes.c_int32),
        ("exact_bids", ctypes.c_uint64),
        ("overbids", ctypes.c_uint64),
        ("underbids", ctypes.c_uint64),
        ("shangas", ctypes.c_uint64),
        ("large_bids_count", ctypes.c_uint64),
        ("large_bids_exact", ctypes.c_uint64),
        ("deals_1_8_count", ctypes.c_uint64),
        ("bids_sum_1_8", ctypes.c_uint64),
        ("tricks_sum_1_8", ctypes.c_uint64),
        ("deals_9_count", ctypes.c_uint64),
        ("bids_sum_9", ctypes.c_uint64),
        ("tricks_sum_9", ctypes.c_uint64),
        ("nines_trump_choices", ctypes.c_uint64),
        ("nines_no_trump_choices", ctypes.c_uint64),
        ("joker_lead_high", ctypes.c_uint64),
        ("joker_lead_low", ctypes.c_uint64),
        ("joker_play_win", ctypes.c_uint64),
        ("joker_play_low", ctypes.c_uint64),
        ("pulkas_played", ctypes.c_uint64),
        ("pulkas_clean", ctypes.c_uint64),
        ("pulkas_bonus_received", ctypes.c_uint64),
        ("pulkas_penalty_received", ctypes.c_uint64),
        ("matches_multi_bonus", ctypes.c_uint64),
    ]


# Сигнатуры одиночной среды
lib.joker_create.argtypes = [ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint64]
lib.joker_create.restype = ctypes.c_void_p
lib.joker_destroy.argtypes = [ctypes.c_void_p]
lib.joker_clone.argtypes = [ctypes.c_void_p]
lib.joker_clone.restype = ctypes.c_void_p
lib.joker_clone_determinized.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint64]
lib.joker_clone_determinized.restype = ctypes.c_void_p
lib.joker_determinize_inplace.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint64]
lib.joker_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint64]
lib.joker_reset256.argtypes = [
    ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint64)
]
lib.joker_force_deck_order.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]

# Observed Mode + Validation + Undo + History + Heuristic
lib.joker_begin_observed_match.argtypes = [
    ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8
]
lib.joker_set_observed_deal.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint8, ctypes.c_uint8]
lib.joker_set_observed_deal.restype = ctypes.c_int
lib.joker_set_observed_remaining_six.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
lib.joker_set_observed_remaining_six.restype = ctypes.c_int

lib.joker_is_done.argtypes = [ctypes.c_void_p]
lib.joker_is_done.restype = ctypes.c_int
lib.joker_get_phase.argtypes = [ctypes.c_void_p]
lib.joker_get_phase.restype = ctypes.c_uint8
lib.joker_get_current_player.argtypes = [ctypes.c_void_p]
lib.joker_get_current_player.restype = ctypes.c_uint8
lib.joker_get_legal_actions_mask.argtypes = [ctypes.c_void_p]
lib.joker_get_legal_actions_mask.restype = ctypes.c_uint64
lib.joker_step.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.joker_step.restype = ctypes.c_int
lib.joker_validate_action.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.joker_validate_action.restype = ctypes.c_int
lib.joker_step_checked.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
lib.joker_step_checked.restype = ctypes.c_int

lib.joker_undo.argtypes = [ctypes.c_void_p]
lib.joker_undo.restype = ctypes.c_int
lib.joker_can_undo_steps.argtypes = [ctypes.c_void_p]
lib.joker_can_undo_steps.restype = ctypes.c_uint32
lib.joker_write_history_tokens.argtypes = [
    ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint8)
]
lib.joker_select_heuristic_action.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
lib.joker_select_heuristic_action.restype = ctypes.c_uint8

lib.joker_tensor_size.restype = ctypes.c_size_t
lib.joker_oracle_tensor_size.restype = ctypes.c_size_t
lib.joker_history_seq_len.restype = ctypes.c_size_t
lib.joker_history_token_dim.restype = ctypes.c_size_t

lib.joker_write_observation_tensor.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_float)]
lib.joker_get_total_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.joker_get_team_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.joker_get_deal_number.argtypes = [ctypes.c_void_p]
lib.joker_get_deal_number.restype = ctypes.c_int32
lib.joker_get_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(JokerSnapshot)]

# Сигнатуры векторизованной среды
lib.joker_vec_create.argtypes = [
    ctypes.c_size_t, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint64, ctypes.c_int
]
lib.joker_vec_create.restype = ctypes.c_void_p
lib.joker_vec_destroy.argtypes = [ctypes.c_void_p]
lib.joker_vec_size.argtypes = [ctypes.c_void_p]
lib.joker_vec_size.restype = ctypes.c_size_t
lib.joker_vec_set_reward_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(JokerRewardConfig)]
lib.joker_vec_set_auto_reset.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.joker_vec_set_stats_enabled.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.joker_vec_reset_stats.argtypes = [ctypes.c_void_p]
lib.joker_vec_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(JokerPlayerStats)]
lib.joker_vec_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint64]
lib.joker_vec_reset_with_seeds.argtypes = [
    ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint64)
]

lib.joker_vec_fork_from_env.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float)
]
lib.joker_vec_get_observations.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint8)
]
lib.joker_vec_step.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint8)
]
lib.joker_vec_advance_to_hero.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint8)
]
lib.joker_vec_step_arena.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint8)
]

# ============================================================================
# 2. Константы и утилиты (каноническая нумерация Action.hpp)
# ============================================================================

RULES_POPULAR, RULES_MINUS, RULES_CLASSIC = 0, 1, 2
MODE_LONG, MODE_SHORT = 0, 1
TEAM_SOLO, TEAM_PAIRS = 0, 1
PULKA_STRICT, PULKA_SOFT = 0, 1

PHASE_TRUMP, PHASE_BIDDING, PHASE_PLAY, PHASE_DONE = 0, 1, 2, 3

SUIT_SPADES, SUIT_CLUBS, SUIT_DIAMONDS, SUIT_HEARTS, SUIT_NONE = 0, 1, 2, 3, 4
RANK_6, RANK_7, RANK_8, RANK_9, RANK_10, RANK_J, RANK_Q, RANK_K, RANK_A = range(9)
RANK_JACK, RANK_QUEEN, RANK_KING, RANK_ACE = RANK_J, RANK_Q, RANK_K, RANK_A

JK1, JK2 = 36, 37
INVALID_CARD = 255

# Канонические ActionId Джокера (синхронизировано с Action.hpp)
ACT_JOKER_PLAY_WIN = 36
ACT_JOKER_PLAY_LOW = 37
ACT_JOKER_LEAD_HIGH_BASE = 38  # 38..41 (+ suit 0..3)
ACT_JOKER_LEAD_LOW_BASE = 42   # 42..45 (+ suit 0..3)

# Коды ошибок JokerStepError
JOKER_ERR_OK = 0
JOKER_ERR_NULL_HANDLE = 1
JOKER_ERR_MATCH_FINISHED = 2
JOKER_ERR_WAITING_FOR_DEAL_INPUT = 3
JOKER_ERR_INVALID_ACTION_ID = 4
JOKER_ERR_FORBIDDEN_DEALER_BID = 5
JOKER_ERR_CARD_NOT_IN_HAND = 6
JOKER_ERR_CARD_ALREADY_PLAYED = 7
JOKER_ERR_CARD_IN_OTHER_KNOWN_HAND = 8
JOKER_ERR_CARD_IS_UPCARD = 9
JOKER_ERR_MUST_FOLLOW_SUIT = 10
JOKER_ERR_MUST_PLAY_TRUMP = 11
JOKER_ERR_MUST_PLAY_HIGHEST = 12
JOKER_ERR_VIOLATES_KNOWN_VOID_OR_BOUND = 13
JOKER_ERR_NO_JOKER_AVAILABLE = 14


def cid(suit: int, rank: int) -> int:
    return suit * 9 + rank


def mask_of(*cards: int) -> int:
    m = 0
    for c in cards:
        m |= (1 << c)
    return m


def legal_actions(mask: int) -> list[int]:
    return [i for i in range(64) if (mask >> i) & 1]


def popcount(mask: int) -> int:
    return int(mask).bit_count()


def make_custom_deck(cards: list[int], deck_size: int = 36) -> ctypes.Array:
    if deck_size == 36:
        all_cards = list(range(1, 9)) + list(range(10, 18)) + list(range(18, 38))
    else:
        all_cards = list(range(38))

    used = set(cards)
    assert len(used) == len(cards), "Дубликаты карт в заданном раскладе!"
    rest = [c for c in all_cards if c not in used]
    full = (cards + rest)[:deck_size]
    while len(full) < 38:
        full.append(INVALID_CARD)
    return (ctypes.c_uint8 * 38)(*full)


def get_snap(env: int) -> JokerSnapshot:
    snap = JokerSnapshot()
    lib.joker_get_snapshot(env, ctypes.byref(snap))
    return snap


def get_obs_vec(env: int, observer: int) -> np.ndarray:
    buf = np.zeros(lib.joker_tensor_size(), dtype=np.float32)
    lib.joker_write_observation_tensor(env, observer, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return buf


def get_history_tokens(env: int, observer: int) -> tuple[np.ndarray, int]:
    seq_len = lib.joker_history_seq_len()
    tok_dim = lib.joker_history_token_dim()
    buf = np.zeros((seq_len, tok_dim), dtype=np.int16)
    length = ctypes.c_uint8(0)
    lib.joker_write_history_tokens(
        env, observer,
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ctypes.byref(length),
    )
    return buf, int(length.value)


def get_scores(env: int) -> list[int]:
    scores = (ctypes.c_int32 * 4)()
    lib.joker_get_total_scores(env, scores)
    return list(scores)


def get_team_scores(env: int) -> list[int]:
    scores = (ctypes.c_int32 * 2)()
    lib.joker_get_team_scores(env, scores)
    return list(scores)


# ============================================================================
# 3. Pytest-фикстуры с автоочисткой памяти
# ============================================================================

if pytest is not None:
    @pytest.fixture
    def make_env():
        created = []

        def _create(rules=RULES_POPULAR, mode=MODE_LONG, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=42):
            env = lib.joker_create(rules, mode, team, pulka, seed)
            assert env is not None
            created.append(env)
            return env

        yield _create

        for env in created:
            lib.joker_destroy(env)

    @pytest.fixture
    def make_vec_env():
        created = []

        def _create(num_envs=16, rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=42, threads=2):
            vec = lib.joker_vec_create(num_envs, rules, mode, team, pulka, seed, threads)
            assert vec is not None
            created.append(vec)
            return vec

        yield _create

        for vec in created:
            lib.joker_vec_destroy(vec)