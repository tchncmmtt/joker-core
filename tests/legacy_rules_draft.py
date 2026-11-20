import ctypes
import os
import platform
import time
from pathlib import Path
import numpy as np

# ============================================================================
# 1. Загрузка скомпилированной библиотеки и описание C ABI
# ============================================================================

def find_library() -> Path:
    root = Path(__file__).resolve().parent
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
# 2. Вспомогательные константы и утилиты
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
    """Формирует буфер из 38 карт, дополняя остаток неиспользованными картами колоды."""
    if deck_size == 36:
        all_cards = (
                list(range(1, 9)) + list(range(10, 18)) + list(range(18, 38))
        )
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


# ============================================================================
# 3. Набор тестов
# ============================================================================

def test_abi_and_dimensions():
    assert lib.joker_tensor_size() == 384, f"Expected 384, got {lib.joker_tensor_size()}"
    assert lib.joker_oracle_tensor_size() == 160, f"Expected 160, got {lib.joker_oracle_tensor_size()}"
    assert lib.joker_history_seq_len() == 48
    assert lib.joker_history_token_dim() == 8
    print("[PASS] ABI и размерности тензоров (384 / 160 / 48x8)")


def test_deck_sizes_and_match_lengths():
    # Popular & Minus -> 36 карт (без 6♠=0 и 6♣=9), Classic -> 38 карт
    for rules, expected_cards in [(RULES_POPULAR, 36), (RULES_MINUS, 36), (RULES_CLASSIC, 38)]:
        env = lib.joker_create(rules, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, 123)
        snap = get_snap(env)
        if rules != RULES_CLASSIC:
            assert snap.phase == PHASE_TRUMP
            lib.joker_step(env, SUIT_SPADES)
            snap = get_snap(env)
        else:
            assert snap.phase == PHASE_BIDDING

        union_hands = snap.hands[0] | snap.hands[1] | snap.hands[2] | snap.hands[3]
        assert popcount(union_hands) == 36
        if rules != RULES_CLASSIC:
            assert (union_hands & (1 << 0)) == 0, "6♠ не должна быть в колоде 36"
            assert (union_hands & (1 << 9)) == 0, "6♣ не должна быть в колоде 36"
        lib.joker_destroy(env)

    # Проверка длины партии Long (24 сдачи) vs Short (8 сдач)
    for mode, expected_deals in [(MODE_LONG, 24), (MODE_SHORT, 8)]:
        env = lib.joker_create(RULES_CLASSIC, mode, TEAM_SOLO, PULKA_STRICT, 777)
        rng = np.random.default_rng(777)
        deals_seen = set()
        steps = 0
        while not lib.joker_is_done(env):
            deals_seen.add(lib.joker_get_deal_number(env))
            acts = legal_actions(lib.joker_get_legal_actions_mask(env))
            assert len(acts) > 0
            ok = lib.joker_step(env, int(rng.choice(acts)))
            assert ok == 1
            steps += 1
        assert len(deals_seen) == expected_deals, f"Ожидалось {expected_deals} сдач, прошло {len(deals_seen)}"
        lib.joker_destroy(env)

    print("[PASS] Размеры колод (36/38) и расписание матчей (Long=24, Short=8)")


def test_visyak_dealer_restriction():
    # В 1-й сдаче Long (1 карта) дилер P0, ходят P1 -> P2 -> P3 -> P0
    env = lib.joker_create(RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 10)
    assert lib.joker_get_current_player(env) == 1
    lib.joker_step(env, 0)  # P1: 0
    lib.joker_step(env, 0)  # P2: 0
    lib.joker_step(env, 0)  # P3: 0 -> сумма 0, карт в сдаче 1 => дилеру P0 запрещено заказывать 1!
    assert lib.joker_get_current_player(env) == 0
    acts = legal_actions(lib.joker_get_legal_actions_mask(env))
    assert acts == [0], f"Дилеру должен быть запрещён висяк 1, доступно: {acts}"

    # Проверяем, что в векторе наблюдения дилера индекс 68 + 1 (69) равен 1.0
    obs = get_obs_vec(env, 0)
    assert obs[69] == 1.0 and obs[68] == 0.0
    lib.joker_destroy(env)
    print("[PASS] Правило «висяка» дилера и его отражение в тензоре")


def test_joker_mechanics():
    # Тест 1: Первый ход Джокером по высшей (LeadHighSuit) на Пики при козыре Червы:
    # У P2 есть пики -> обязан положить старшую пику. У P3 нет пик, но есть козырь -> обязан бить козырем и забирает взятку!
    env = lib.joker_create(RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 1)
    # 2-я сдача (2 карты каждому)
    deck = make_custom_deck([
        cid(SUIT_CLUBS, RANK_7), cid(SUIT_CLUBS, RANK_8),       # P0 (дилер)
        JK1, cid(SUIT_CLUBS, RANK_9),                           # P1 (ходит первым)
        cid(SUIT_SPADES, RANK_8), cid(SUIT_SPADES, RANK_ACE),   # P2 (имеет две пики -> обязан дать A♠)
        cid(SUIT_HEARTS, RANK_6), cid(SUIT_DIAMONDS, RANK_ACE), # P3 (нет пик, есть козырь 6♥)
        cid(SUIT_HEARTS, RANK_KING)                             # upcard -> козырь Червы (♥)
    ])
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 1)

    # Переходим ко 2-й сдаче (проматываем 1-ю сдачу)
    while lib.joker_get_deal_number(env) == 0:
        acts = legal_actions(lib.joker_get_legal_actions_mask(env))
        lib.joker_step(env, acts[0])

    # Теперь сдача 1 (по 2 карты), дилер P1, ходит P2 -> P3 -> P0 -> P1.
    # Зададим расклад прямо на 1-й сдаче с 1 картой для чистоты!
    lib.joker_destroy(env)

    env = lib.joker_create(RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 1)
    deck1 = make_custom_deck([
        cid(SUIT_DIAMONDS, RANK_ACE), # P0 (дилер)
        JK1,                          # P1 (ходит первым)
        cid(SUIT_SPADES, RANK_ACE),   # P2 (имеет пику -> обязан дать A♠)
        cid(SUIT_HEARTS, RANK_6),     # P3 (нет пик -> обязан дать козырь 6♥ и забрать взятку!)
        cid(SUIT_HEARTS, RANK_KING)   # upcard -> козырь Червы
    ])
    lib.joker_force_deck_order(env, deck1)
    lib.joker_reset(env, RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 1)

    # Торговля: P1=0, P2=0, P3=1, P0=1 (висяк 0 запрещён)
    lib.joker_step(env, 0)
    lib.joker_step(env, 0)
    lib.joker_step(env, 1)
    lib.joker_step(env, 1)

    # Розыгрыш: P1 ходит JK1 по высшей Пике (action 36 + SUIT_SPADES = 36)
    assert lib.joker_get_current_player(env) == 1
    assert lib.joker_step(env, 36 + SUIT_SPADES) == 1
    assert lib.joker_step(env, cid(SUIT_SPADES, RANK_ACE)) == 1
    assert lib.joker_step(env, cid(SUIT_HEARTS, RANK_6)) == 1
    assert lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_ACE)) == 1

    # Сдача завершилась, P3 взял взятку и выполнил заказ 1 => +100 (в Popular за 1/1 из 1 карты = 100*1 = 100)
    scores = (ctypes.c_int32 * 4)()
    lib.joker_get_total_scores(env, scores)
    assert scores[3] == 100, f"Ожидалось 100 очков у P3 (перебил Джокера козырем), получено {scores[3]}"
    lib.joker_destroy(env)

    # Тест 2: Два Джокера на выигрыш (PlayWin=44) во взятке -> забирает второй!
    env2 = lib.joker_create(RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 2)
    deck2 = make_custom_deck([
        cid(SUIT_DIAMONDS, RANK_6),   # P0 (дилер)
        cid(SUIT_SPADES, RANK_7),     # P1 (ходит первым 7♠)
        JK1,                          # P2 (бьёт первым Джокером PlayWin=44)
        JK2,                          # P3 (бьёт вторым Джокером PlayWin=44 -> забирает!)
        cid(SUIT_HEARTS, RANK_KING)   # upcard -> козырь Червы
    ])
    lib.joker_force_deck_order(env2, deck2)
    lib.joker_reset(env2, RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 2)
    lib.joker_step(env2, 0)  # P1
    lib.joker_step(env2, 0)  # P2
    lib.joker_step(env2, 1)  # P3
    lib.joker_step(env2, 1)  # P0

    lib.joker_step(env2, cid(SUIT_SPADES, RANK_7))  # P1
    lib.joker_step(env2, 44)                        # P2: JK1(Win)
    lib.joker_step(env2, 44)                        # P3: JK2(Win) -> побеждает
    lib.joker_step(env2, cid(SUIT_DIAMONDS, RANK_6))# P0

    scores2 = (ctypes.c_int32 * 4)()
    lib.joker_get_total_scores(env2, scores2)
    assert scores2[3] == 100 and scores2[2] == 50, f"Второй Джокер должен побеждать: {list(scores2)}"
    lib.joker_destroy(env2)

    print("[PASS] Механика Джокеров (LeadHighSuit + козырь, второй PlayWin побеждает)")


def test_scoring_presets_popular_minus_classic():
    # Проверяем формулу "все свои" и "штангу" во всех 3 пресетах
    for rules, expected_all_own, expected_shanga in [
        (RULES_POPULAR, 100, -200),
        (RULES_MINUS,   100, -100),
        (RULES_CLASSIC, 100,    0),
    ]:
        env = lib.joker_create(rules, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 5)
        deck = make_custom_deck([
            cid(SUIT_DIAMONDS, RANK_6),   # P0 (дилер)
            cid(SUIT_SPADES, RANK_ACE),   # P1 (возьмёт взятку)
            cid(SUIT_SPADES, RANK_7),     # P2 (закажет 1 и попадёт в штангу)
            cid(SUIT_DIAMONDS, RANK_7),   # P3 (закажет 0 и возьмёт пас = +50)
            cid(SUIT_HEARTS, RANK_KING)   # upcard -> козырь Червы
        ], deck_size=38 if rules == RULES_CLASSIC else 36)
        lib.joker_force_deck_order(env, deck)
        lib.joker_reset(env, rules, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 5)

        lib.joker_step(env, 1)  # P1: 1/1 ("все свои")
        lib.joker_step(env, 1)  # P2: 1/0 ("штанга")
        lib.joker_step(env, 0)  # P3: 0/0 (пас)
        lib.joker_step(env, 0)  # P0: 0/0 (пас, сумма=2 != 1)

        lib.joker_step(env, cid(SUIT_SPADES, RANK_ACE))
        lib.joker_step(env, cid(SUIT_SPADES, RANK_7))
        lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_7))
        lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_6))

        scores = (ctypes.c_int32 * 4)()
        lib.joker_get_total_scores(env, scores)
        assert scores[1] == expected_all_own, f"Rules {rules}: P1={scores[1]} != {expected_all_own}"
        assert scores[2] == expected_shanga, f"Rules {rules}: P2={scores[2]} != {expected_shanga}"
        assert scores[3] == 50 and scores[0] == 50
        lib.joker_destroy(env)

    print("[PASS] Подсчёт очков в пресетах Popular / Minus / Classic")


def run_short_pulka_scenario(team_mode: int, pulka_mode: int, clean_players: set[int]) -> list[int]:
    """
    Проигрывает 1-ю пульку (4 сдачи по 9 карт) в Classic + Short.
    В первых 3 сдачах все заказывают и берут по 1 взятке (кроме P0: 6 взяток, чтобы сумма=9 -> дилер берет 0).
    На самом деле проще:
    - В сдачах 0, 1, 2: P0 берёт 3 взятки (заказ 3 -> +200), P1, P2, P3 берут по 2 взятки (заказ 2 -> +150),
      а последний дилер меняет ставку так, чтобы избежать висяка, но при этом для чистоты:
      пусть P0 берёт 3 (+200), P1 берёт 3 (+200), P2 берёт 2 (+150), P3 берёт 1 (+100).
      Так как дилер в каждой сдаче разный, просто заставим каждого в сдачах 0..2 выполнить свой контракт:
      сдача 0 (дилер P0): P1=3/3(+200), P2=3/3(+200), P3=2/2(+150), P0=1/1(+100) [сумма 9, висяк P0=0, P0 ставит 1]
      сдача 1 (дилер P1): P2=3/3(+200), P3=3/3(+200), P0=2/2(+150), P1=1/1(+100)
      сдача 2 (дилер P2): P3=3/3(+200), P0=3/3(+200), P1=2/2(+150), P2=1/1(+100)
      Итог за первые 3 сдачи: у КАЖДОГО игрока есть записи [100, 150, 200] (сумма 450, макс без последней = 200),
      и все четверо пока идут чисто!
    - В 4-й сдаче (дилер P3, последняя сдача пульки):
      каждый из `clean_players` выполняет заказ (1 или 2 взятки), а остальные умышленно срываются!
    """
    env = lib.joker_create(RULES_CLASSIC, MODE_SHORT, team_mode, pulka_mode, 42)

    # Помощник для проведения одной 9-карточной сдачи с заданными взятками для P0..P3
    def play_9_deal(target_tricks: list[int], exact_mask: list[bool]):
        assert sum(target_tricks) == 9
        snap = get_snap(env)
        dealer = snap.dealer
        # Раздаём карты так, чтобы P0..P3 забрали ровно target_tricks[p] взяток:
        # P0 получает старшие Пики, P1 — старшие Трефы, P2 — старшие Бубны, P3 — старшие Червы,
        # а upcard = JK1 (бескозырка), чтобы старшая карта масти первого хода всегда брала взятку!
        pools = [
            [cid(SUIT_SPADES, r) for r in range(8, -1, -1)],
            [cid(SUIT_CLUBS, r) for r in range(8, -1, -1)],
            [cid(SUIT_DIAMONDS, r) for r in range(8, -1, -1)],
            [cid(SUIT_HEARTS, r) for r in range(8, -1, -1)],
        ]
        # Используем бескозырку: каждый победитель взятки заходит своей мастью и забирает!
        # Чтобы P_k забрал ровно target_tricks[k] взяток, даём каждому P_k:
        # target_tricks[k] тузов/королей своей масти (которые гарантированно бьют) + (9 - target_tricks[k]) мусорных карт
        winners_seq = []
        for p in range(4):
            winners_seq.extend([p] * target_tricks[p])

        # Формируем руки по 9 карт:
        hands = [[] for _ in range(4)]
        high_idx = [0, 0, 0, 0]
        low_idx = [8, 8, 8, 8]

        for p in range(4):
            for _ in range(target_tricks[p]):
                hands[p].append(pools[p][high_idx[p]])
                high_idx[p] += 1

        # Добиваем остаток рук неиспользованными картами из пулов
        rem = []
        for p in range(4):
            rem.extend(pools[p][high_idx[p]:low_idx[p] + 1])
        rem.extend([JK1, JK2])

        ptr = 0
        for p in range(4):
            while len(hands[p]) < 9:
                hands[p].append(rem[ptr])
                ptr += 1

        # Последняя 37-я карта (индекс 36) = rem[ptr] -> убедимся что это Джокер (бескозырка!)
        upcard = JK1 if JK1 not in sum(hands, []) else JK2
        # Уберём оба Джокера из рук в запас, чтобы upcard гарантированно был JK1
        clean_rem = [c for p in range(4) for c in pools[p][high_idx[p]:low_idx[p] + 1]]
        hands = [[] for _ in range(4)]
        high_idx = [0, 0, 0, 0]
        for p in range(4):
            for _ in range(target_tricks[p]):
                hands[p].append(pools[p][high_idx[p]])
                high_idx[p] += 1
        ptr = 0
        for p in range(4):
            while len(hands[p]) < 9:
                hands[p].append(clean_rem[ptr])
                ptr += 1
        upcard = JK1

        deck_arr = (ctypes.c_uint8 * 38)(*(sum(hands, []) + [upcard, JK2]))
        lib.joker_force_deck_order(env, deck_arr)
        # Перезапускаем текущую сдачу с нужным дилером через клонирование/шаги не нужно:
        # ForceDeckOrder сработает на следующей сдаче, если вызвать перед BeginDeal,
        # но для сдачи 0 мы можем сделать reset!

    lib.joker_destroy(env)

    # Более простой и наглядный способ: в 1-й сдаче Short (9 карт) сразу проверить расчёт пульки
    # невозможно, так как пулька в Short длится 4 сдачи. Прогоним все 4 сдачи на бескозырке!
    env = lib.joker_create(RULES_CLASSIC, MODE_SHORT, team_mode, pulka_mode, 99)

    def setup_deal_deck(target_tricks: list[int]):
        pools = [
            [cid(SUIT_SPADES, r) for r in range(8, -1, -1)],
            [cid(SUIT_CLUBS, r) for r in range(8, -1, -1)],
            [cid(SUIT_DIAMONDS, r) for r in range(8, -1, -1)],
            [cid(SUIT_HEARTS, r) for r in range(8, -1, -1)],
        ]
        hands = [[] for _ in range(4)]
        high_idx = [0, 0, 0, 0]
        for p in range(4):
            for _ in range(target_tricks[p]):
                hands[p].append(pools[p][high_idx[p]])
                high_idx[p] += 1
        clean_rem = [c for p in range(4) for c in pools[p][high_idx[p]:]]
        ptr = 0
        for p in range(4):
            while len(hands[p]) < 9:
                hands[p].append(clean_rem[ptr])
                ptr += 1
        deck_arr = (ctypes.c_uint8 * 38)(*(sum(hands, []) + [JK1, JK2]))
        lib.joker_force_deck_order(env, deck_arr)

    # Сдачи 0, 1, 2: все 4 игрока берут строго свои заказы, максимальная запись у каждого = 200
    schedules = [
        [1, 3, 3, 2],  # сдача 0 (дилер P0): P1=3, P2=3, P3=2, P0=1 (висяк 0, P0 ставит 1 -> ок!)
        [2, 1, 3, 3],  # сдача 1 (дилер P1): P2=3, P3=3, P0=2, P1=1 (висяк 0, P1 ставит 1 -> ок!)
        [3, 2, 1, 3],  # сдача 2 (дилер P2): P3=3, P0=3, P1=2, P2=1 (висяк 0, P2 ставит 1 -> ок!)
    ]

    # Сдача 3 (финальная сдача 1-й пульки, дилер P3):
    # Выбирает, кто выйдет чисто, а кто сорвётся
    # Если игрок в clean_players -> ставит 2 и берёт 2 (+150). Если нет -> ставит 3, но берёт меньше (срыв).
    t3 = [2 if p in clean_players else 0 for p in range(4)]
    # Добиваем сумму взяток в 4-й сдаче до 9 за счёт невышедших игроков
    rem_tricks = 9 - sum(t3)
    for p in range(4):
        if p not in clean_players:
            add = min(rem_tricks, 4)
            t3[p] += add
            rem_tricks -= add
    if rem_tricks > 0:
        # На случай если все 4 в clean_players (sum=8): P0 возьмёт 3 и закажет 3
        for p in range(4):
            if p in clean_players:
                t3[p] += rem_tricks
                break

    schedules.append(t3)

    # Применяем расклад 0-й сдачи через reset
    setup_deal_deck(schedules[0])
    lib.joker_reset(env, RULES_CLASSIC, MODE_SHORT, team_mode, pulka_mode, 99)

    for deal_idx in range(4):
        # Перед завершением текущей сдачи задаём колоду для следующей сдачи
        if deal_idx + 1 < 4:
            setup_deal_deck(schedules[deal_idx + 1])

        snap = get_snap(env)
        assert snap.deal_number == deal_idx
        assert snap.trump_suit == SUIT_NONE

        # Торговля
        for _ in range(4):
            cur = lib.joker_get_current_player(env)
            mask = lib.joker_get_legal_actions_mask(env)
            acts = legal_actions(mask)
            if deal_idx < 3 or cur in clean_players:
                desired_bid = schedules[deal_idx][cur]
            else:
                # Умышленно срываем заказ (ставим 0 при взятых >0, либо максимум)
                desired_bid = 0 if schedules[deal_idx][cur] > 0 else 5
                if desired_bid not in acts:
                    desired_bid = acts[-1] if acts[-1] != schedules[deal_idx][cur] else acts[0]
            assert desired_bid in acts, f"Deal {deal_idx}, P{cur}: bid {desired_bid} not in {acts}"
            lib.joker_step(env, desired_bid)

        # Розыгрыш 9 взяток:
        # Когда ходит игрок p, у которого ещё есть старшая карта его масти (из целевых взяток), он ходит ею и забирает.
        # Если у лидера взятки старших карт не осталось, он сбрасывает карту другой масти, а тот, у кого есть туз этой масти, забирает!
        while lib.joker_get_deal_number(env) == deal_idx and not lib.joker_is_done(env):
            cur = lib.joker_get_current_player(env)
            snap = get_snap(env)
            acts = legal_actions(lib.joker_get_legal_actions_mask(env))
            # Берём старшую легальную карту, если первый ход и мы ещё не добрали target_tricks, иначе младшую
            if snap.cards_in_trick == 0 and snap.tricks_taken[cur] < schedules[deal_idx][cur]:
                # Ищем карту своей "сильной" масти (масть == cur)
                own_suit_cards = [a for a in acts if a < 36 and (a // 9) == cur]
                chosen = max(own_suit_cards) if own_suit_cards else max(acts)
            else:
                # Сбрасываем младшую карту не своей сильной масти, если возможно
                other_cards = [a for a in acts if a < 36 and (a // 9) != cur]
                chosen = min(other_cards) if other_cards else min(acts)
            lib.joker_step(env, chosen)

    scores = (ctypes.c_int32 * 4)()
    lib.joker_get_total_scores(env, scores)
    lib.joker_destroy(env)
    return list(scores)


def test_pulka_strict_vs_soft_solo_and_pairs():
    # У каждого игрока за первые 3 сдачи набрано 450 очков (записи 100, 150, 200 -> макс запись = 200).
    # В 4-й сдаче:
    # - вышедший игрок берёт 2/2 (+150 -> итог до пульки = 600).
    # - невышедший игрок берёт >0 при заказе 0 (перебор -> +10*взяток).

    # 1. SOLO + STRICT:
    # Вышел один P0 -> P0 получает +200 (итог 800), у P1, P2, P3 списывается -200!
    s_solo_strict_1 = run_short_pulka_scenario(TEAM_SOLO, PULKA_STRICT, {0})
    assert s_solo_strict_1[0] == 800, f"Solo Strict (1 clean): P0={s_solo_strict_1[0]} != 800"
    assert s_solo_strict_1[1] < 450 and s_solo_strict_1[2] < 450 and s_solo_strict_1[3] < 450

    # Вышли двое (P0 и P2) -> в Solo Strict оба сжигают бонус друг другу (P0=600, P2=600), а у P1 и P3 списывается -200!
    s_solo_strict_2 = run_short_pulka_scenario(TEAM_SOLO, PULKA_STRICT, {0, 2})
    assert s_solo_strict_2[0] == 600 and s_solo_strict_2[2] == 600, f"Solo Strict (2 clean): {s_solo_strict_2}"

    # 2. PAIRS + STRICT:
    # Вышли оба партнёра (P0 и P2) -> в Pairs Strict партнёры НЕ сжигают бонус друг другу: оба получают +200 (800, 800)!
    s_pairs_strict_team = run_short_pulka_scenario(TEAM_PAIRS, PULKA_STRICT, {0, 2})
    assert s_pairs_strict_team[0] == 800 and s_pairs_strict_team[2] == 800, f"Pairs Strict (both partners): {s_pairs_strict_team}"

    # Вышел только P0 -> свой своего не штрафует: P2 (напарник) НЕ получает -200 (остаётся >450), а враги P1 и P3 получают -200!
    s_pairs_strict_one = run_short_pulka_scenario(TEAM_PAIRS, PULKA_STRICT, {0})
    assert s_pairs_strict_one[0] == 800
    assert s_pairs_strict_one[2] > 450, f"Pairs Strict: напарник P2 не должен штрафоваться ({s_pairs_strict_one})"
    assert s_pairs_strict_one[1] < 450 and s_pairs_strict_one[3] < 450

    # 3. SOFT PULKA MODE (и для Solo, и для Pairs):
    # Сосед "над" P0 — это P1 ((0+1)%4). Сосед "под" P0 — это P3 ((0+3)%4).
    # Если вышли P0 и P2 (сидят напротив):
    # - Над P0 сидит P1 (не вышел) -> P0 ПОЛУЧАЕТ +200 (800)!
    # - Над P2 сидит P3 (не вышел) -> P2 ПОЛУЧАЕТ +200 (800)!
    # - P1 сидит под P2 (вышел) -> у P1 списывается -200!
    # - P3 сидит под P0 (вышел) -> у P3 списывается -200!
    for t_mode in (TEAM_SOLO, TEAM_PAIRS):
        s_soft_opp = run_short_pulka_scenario(t_mode, PULKA_SOFT, {0, 2})
        assert s_soft_opp[0] == 800 and s_soft_opp[2] == 800, f"Soft ({t_mode}) P0 & P2 clean: {s_soft_opp}"
        assert s_soft_opp[1] < 450 and s_soft_opp[3] < 450

    # Если вышли P0 и P1 (P1 сидит НАД P0):
    # - Над P0 сидит P1 (вышел!) -> у P0 премия СГОРАЕТ (600)!
    # - Над P1 сидит P2 (не вышел) -> P1 ПОЛУЧАЕТ +200 (800)!
    # - Под P0 сидит P3 -> у P3 списывается -200!
    # - Под P1 сидит P0 (но P0 сам вышел чисто -> списания нет, но и бонуса нет -> 600)!
    # - Над P2 сидит P3 (не вышел) -> у P2 списания НЕТ (>450)!
    s_soft_adj = run_short_pulka_scenario(TEAM_SOLO, PULKA_SOFT, {0, 1})
    assert s_soft_adj[0] == 600, f"Soft: P1 над P0 должен сжечь бонус P0, получено {s_soft_adj}"
    assert s_soft_adj[1] == 800, f"Soft: P1 должен получить бонус, получено {s_soft_adj}"
    assert s_soft_adj[2] > 450, f"Soft: P2 не сидит под вышедшим, штрафа быть не должно: {s_soft_adj}"
    assert s_soft_adj[3] < 450, f"Soft: P3 сидит под P0, должен получить штраф: {s_soft_adj}"

    print("[PASS] Расчёт пульки: Solo/Pairs × Strict/Soft проверены на всех граничных случаях")


def test_determinization_respects_void_suits():
    # Проверяем, что детерминизация для IS-MCTS не раздаёт оппоненту масть, в которой у него зафиксирован ренонс
    env = lib.joker_create(RULES_CLASSIC, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, 123)
    # Делаем 3-карточную сдачу (3-я сдача в Long)
    lib.joker_destroy(env)

    env = lib.joker_create(RULES_CLASSIC, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 123)
    # 3-я сдача (deal_number=2, 3 карты каждому, дилер P2, ходит P3 -> P0 -> P1 -> P2)
    deck = make_custom_deck([
        cid(SUIT_CLUBS, RANK_7), cid(SUIT_CLUBS, RANK_8), cid(SUIT_CLUBS, RANK_9),       # P0
        cid(SUIT_DIAMONDS, RANK_7), cid(SUIT_DIAMONDS, RANK_8), cid(SUIT_DIAMONDS, RANK_9), # P1 (НЕТ ПИК!)
        cid(SUIT_HEARTS, RANK_7), cid(SUIT_HEARTS, RANK_8), cid(SUIT_HEARTS, RANK_9),     # P2
        cid(SUIT_SPADES, RANK_ACE), cid(SUIT_SPADES, RANK_KING), cid(SUIT_SPADES, RANK_QUEEN), # P3
        cid(SUIT_HEARTS, RANK_ACE) # upcard = Червы
    ], deck_size=38)
    lib.joker_force_deck_order(env, deck)

    # Доходим до deal_number == 2
    while lib.joker_get_deal_number(env) < 2:
        if lib.joker_get_deal_number(env) == 1:
            lib.joker_force_deck_order(env, deck)
        acts = legal_actions(lib.joker_get_legal_actions_mask(env))
        lib.joker_step(env, acts[0])

    # Торговля в 3-й сдаче
    for _ in range(4):
        acts = legal_actions(lib.joker_get_legal_actions_mask(env))
        lib.joker_step(env, acts[0])

    # Ходит P3 тузом пик (A♠), P0 сносит 7♣ (ренонс в пиках и козыре!), P1 сносит 7♦ (ренонс в пиках и козыре!)
    lib.joker_step(env, cid(SUIT_SPADES, RANK_ACE)) # P3
    lib.joker_step(env, cid(SUIT_CLUBS, RANK_7))    # P0 -> void Spades & Hearts
    lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_7)) # P1 -> void Spades & Hearts
    lib.joker_step(env, cid(SUIT_HEARTS, RANK_7))   # P2 -> козыряет 7♥ (void Spades)

    # Проверяем в векторе наблюдения P3, что ренонсы отразились в индексах 348..359
    obs_p3 = get_obs_vec(env, 3)
    # rel=1 (P0): void Spades (348)=1, void Hearts (351)=1
    assert obs_p3[348] == 1.0 and obs_p3[351] == 1.0

    # Делаем 30 независимых детерминизаций от лица P3 и проверяем, что ни P0, ни P1, ни P2 не получили Пики!
    spades_mask = sum(1 << cid(SUIT_SPADES, r) for r in range(9))
    hearts_mask = sum(1 << cid(SUIT_HEARTS, r) for r in range(9))
    for seed in range(1, 31):
        det = lib.joker_clone_determinized(env, 3, seed)
        snap = get_snap(det)
        assert (snap.hands[0] & spades_mask) == 0 and (snap.hands[0] & hearts_mask) == 0
        assert (snap.hands[1] & spades_mask) == 0 and (snap.hands[1] & hearts_mask) == 0
        assert (snap.hands[2] & spades_mask) == 0
        lib.joker_destroy(det)

    lib.joker_destroy(env)
    print("[PASS] Честная детерминизация IS-MCTS с соблюдением ренонсов (voidSuits)")


def test_vector_env_oracle_tokens_and_pairs_rewards():
    num_envs = 32
    obs_dim = lib.joker_tensor_size()
    oracle_dim = lib.joker_oracle_tensor_size()
    seq_len = lib.joker_history_seq_len()
    tok_dim = lib.joker_history_token_dim()

    vec = lib.joker_vec_create(num_envs, RULES_POPULAR, MODE_SHORT, TEAM_PAIRS, PULKA_SOFT, 2026, 4)
    assert lib.joker_vec_size(vec) == num_envs

    obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
    masks = np.zeros(num_envs, dtype=np.uint64)
    players = np.zeros(num_envs, dtype=np.uint8)
    oracle = np.zeros((num_envs, oracle_dim), dtype=np.float32)
    tokens = np.zeros((num_envs, seq_len, tok_dim), dtype=np.int16)
    lengths = np.zeros(num_envs, dtype=np.uint8)

    lib.joker_vec_get_observations(
        vec,
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        players.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        oracle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )

    assert not np.isnan(obs).any()
    assert not np.isnan(oracle).any()
    assert np.all(lengths == 0)
    assert np.all(tokens == 0)

    # Прогоняем 120 шагов и проверяем награды в Pairs (должны быть строго симметричны: r[0]==r[2]==-r[1]==-r[3])
    rewards = np.zeros((num_envs, 4), dtype=np.float32)
    deal_dones = np.zeros(num_envs, dtype=np.uint8)
    match_dones = np.zeros(num_envs, dtype=np.uint8)
    final_scores = np.zeros((num_envs, 4), dtype=np.int32)
    rng = np.random.default_rng(123)

    for step in range(120):
        actions = np.array([rng.choice(legal_actions(int(m))) for m in masks], dtype=np.uint8)
        lib.joker_vec_step(
            vec,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            players.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            deal_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            match_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            final_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            oracle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        # Проверка симметрии наград команд
        assert np.allclose(rewards[:, 0], rewards[:, 2], atol=1e-6)
        assert np.allclose(rewards[:, 1], rewards[:, 3], atol=1e-6)
        assert np.allclose(rewards[:, 0], -rewards[:, 1], atol=1e-6)

        # Проверка корректности токенов истории (0 = PAD за пределами length[i], >0 внутри)
        for i in range(num_envs):
            L = int(lengths[i])
            if L > 0:
                assert np.all(tokens[i, :L, 0] >= 1) and np.all(tokens[i, :L, 0] <= 3)
                assert np.all(tokens[i, :L, 1] >= 1) and np.all(tokens[i, :L, 1] <= 4)
            if L < seq_len:
                assert np.all(tokens[i, L:, :] == 0)

    # Проверка ForkFromEnv для гибридного поиска
    root_env = lib.joker_create(RULES_CLASSIC, MODE_SHORT, TEAM_PAIRS, PULKA_SOFT, 555)
    lib.joker_vec_fork_from_env(
        vec,
        root_env,
        0,
        999,
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        oracle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    # У observer=0 своя рука во всех N слотах обязана совпадать!
    my_hand_0 = oracle[0, :38]
    for i in range(1, num_envs):
        assert np.array_equal(oracle[i, :38], my_hand_0)

    lib.joker_destroy(root_env)
    lib.joker_vec_destroy(vec)
    print("[PASS] Векторная среда: Pairs-награды, Oracle (160), Transformer-токены (48x8) и ForkFromEnv")


def benchmark_throughput(num_envs: int = 1024, num_steps: int = 2000):
    obs_dim = lib.joker_tensor_size()
    vec = lib.joker_vec_create(num_envs, RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 42, 0)

    obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
    masks = np.zeros(num_envs, dtype=np.uint64)
    players = np.zeros(num_envs, dtype=np.uint8)
    rewards = np.zeros((num_envs, 4), dtype=np.float32)
    deal_dones = np.zeros(num_envs, dtype=np.uint8)
    match_dones = np.zeros(num_envs, dtype=np.uint8)
    final_scores = np.zeros((num_envs, 4), dtype=np.int32)

    lib.joker_vec_get_observations(
        vec,
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        players.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        None, None, None,
    )

    # Быстрый выбор первого легального бита (trailing zeros) через NumPy
    t0 = time.perf_counter()
    total_matches = 0
    for _ in range(num_steps):
        lsb = masks & (~masks + np.uint64(1))
        actions = np.log2(lsb).astype(np.uint8)

        lib.joker_vec_step(
            vec,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            players.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            deal_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            match_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            final_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            None, None, None,
        )
        total_matches += int(match_dones.sum())

    dt = time.perf_counter() - t0
    total_steps = num_envs * num_steps
    print(
        f"[BENCH] {num_envs} сред × {num_steps} шагов = {total_steps:,} шагов за {dt:.3f} с "
        f"-> {total_steps / dt:,.0f} шагов/сек ({total_matches:,} завершённых матчей)"
    )
    lib.joker_vec_destroy(vec)


if __name__ == "__main__":
    print(f"Загружена библиотека: {LIB_PATH}")
    test_abi_and_dimensions()
    test_deck_sizes_and_match_lengths()
    test_visyak_dealer_restriction()
    test_joker_mechanics()
    test_scoring_presets_popular_minus_classic()
    test_pulka_strict_vs_soft_solo_and_pairs()
    test_determinization_respects_void_suits()
    test_vector_env_oracle_tokens_and_pairs_rewards()
    benchmark_throughput(num_envs=1024, num_steps=2000)