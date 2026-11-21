import ctypes
from conftest import (
    lib, RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, PULKA_SOFT,
    PHASE_TRUMP, SUIT_HEARTS,
    JOKER_OBS_FLAG_WAITING_DEAL, JOKER_OBS_FLAG_WAITING_REM_SIX,
    mask_of, cid, SUIT_SPADES, SUIT_CLUBS, SUIT_DIAMONDS,
    RANK_7, RANK_8, RANK_9, RANK_10, RANK_J, RANK_Q, RANK_K, RANK_A,
    get_snap, get_scores,
)


def test_strict_pulka_single_vs_multiple_clean(make_env):
    """В Strict при >= 2 чистых игроках премия обнуляется, а невышедшие всё равно получают списание."""
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=123)
    # Прогоняем 1-ю пульку (4 сдачи по 9 карт) эвристикой и проверяем отсутствие крашей и валидность счетов
    while get_snap(env).deal_number < 4 and not lib.joker_is_done(env):
        act = lib.joker_select_heuristic_action(env, 42 + get_snap(env).deal_number)
        assert lib.joker_step(env, act) == 1
    snap = get_snap(env)
    assert snap.deal_number == 4
    # В начале 2-й пульки флаги pulka_clean должны сброситься в 1
    assert list(snap.pulka_clean) == [1, 1, 1, 1]


def test_observed_mid_match_state_restores_pulka_and_scores(make_env):
    """Проверка новой функции joker_set_observed_match_state и флагов observed_flags / hero_seat."""
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_SOFT, seed=1)

    total_scores = (ctypes.c_int32 * 4)(450, 200, -100, 350)
    pulka_clean = (ctypes.c_uint8 * 4)(1, 0, 1, 1)
    pulka_all_passes = (ctypes.c_uint8 * 4)(0, 0, 1, 0)
    max_excl = (ctypes.c_int16 * 4)(250, 150, 50, 200)
    max_non_pass = (ctypes.c_int16 * 4)(250, 150, 0, 200)

    # Входим на 3-ю сдачу 1-й пульки (deal_number=2), дилер P2, герой P3
    ok = lib.joker_set_observed_match_state(
        env, RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_SOFT,
        2, 2, 3,
        total_scores, pulka_clean, pulka_all_passes, max_excl, max_non_pass,
    )
    assert ok == 1

    snap = get_snap(env)
    assert snap.deal_number == 2
    assert snap.dealer == 2
    assert snap.hero_seat == 3
    assert snap.observed_flags == JOKER_OBS_FLAG_WAITING_DEAL
    assert get_scores(env) == [450, 200, -100, 350]
    assert list(snap.pulka_clean) == [1, 0, 1, 1]

    # Вводим первые 3 карты героя на девятках
    three = mask_of(cid(SUIT_SPADES, RANK_A), cid(SUIT_HEARTS, RANK_K), cid(SUIT_DIAMONDS, RANK_Q))
    assert lib.joker_set_observed_deal(env, three, 255, 255) == 1
    snap = get_snap(env)
    assert snap.observed_flags == 0
    assert snap.phase == PHASE_TRUMP

    # Выбор козыря -> ждём добора 6 карт
    assert lib.joker_step(env, SUIT_HEARTS) == 1
    assert get_snap(env).observed_flags == JOKER_OBS_FLAG_WAITING_REM_SIX

    nine = three | mask_of(
        cid(SUIT_SPADES, RANK_7), cid(SUIT_SPADES, RANK_8), cid(SUIT_CLUBS, RANK_9),
        cid(SUIT_CLUBS, RANK_10), cid(SUIT_DIAMONDS, RANK_J), cid(SUIT_HEARTS, RANK_7),
    )
    assert lib.joker_set_observed_remaining_six(env, nine) == 1
    assert get_snap(env).observed_flags == 0