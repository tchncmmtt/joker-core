import ctypes
import platform
from pathlib import Path
import numpy as np
import pytest

# ============================================================================
# 1. Загрузка библиотеки и описание C ABI
# ============================================================================

def find_library() -> Path:
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
lib.joker_vec_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint64]

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
# 2. Константы и утилиты
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


def cid(suit: int, rank: int) -> int:
    return suit * 9 + rank


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


def get_scores(env: int) -> list[int]:
    scores = (ctypes.c_int32 * 4)()
    lib.joker_get_total_scores(env, scores)
    return list(scores)


# ============================================================================
# 3. Pytest-фикстуры с автоочисткой памяти
# ============================================================================

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
