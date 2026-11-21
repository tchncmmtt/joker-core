import numpy as np
from conftest import (
    lib, RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_STRICT,
    PHASE_TRUMP, PHASE_BIDDING, PHASE_PLAY,
    get_snap, get_obs_vec, get_history_tokens,
)


def test_history_tokens_record_all_phases_and_clear_on_new_deal(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=42)

    # 1. Выбор козыря (eventType = 1)
    assert get_snap(env).phase == PHASE_TRUMP
    act_trump = lib.joker_select_heuristic_action(env, 1)
    assert lib.joker_step(env, act_trump) == 1
    toks, length = get_history_tokens(env, 1)
    assert length == 1
    assert toks[0, 0] == 1  # ChoosingTrump
    assert toks[0, 2] == act_trump + 1
    assert np.all(toks[1:] == 0)  # PAD = 0

    # 2. 4 заказа (eventType = 2)
    for i in range(4):
        assert get_snap(env).phase == PHASE_BIDDING
        act_bid = lib.joker_select_heuristic_action(env, 10 + i)
        assert lib.joker_step(env, act_bid) == 1

    toks, length = get_history_tokens(env, 1)
    assert length == 5
    assert list(toks[1:5, 0]) == [2, 2, 2, 2]

    # 3. Первый ход во взятке (eventType = 3)
    assert get_snap(env).phase == PHASE_PLAY
    act_card = lib.joker_select_heuristic_action(env, 99)
    assert lib.joker_step(env, act_card) == 1
    toks, length = get_history_tokens(env, 1)
    assert length == 6
    assert toks[5, 0] == 3  # TrickPlay
    assert toks[5, 6] == 1  # trickIdx + 1 == 1
    assert toks[5, 7] == 1  # posInTrick + 1 == 1

    # 4. Доигрываем сдачу до конца -> буфер истории должен очиститься
    step_i = 0
    while get_snap(env).deal_number == 0:
        act = lib.joker_select_heuristic_action(env, 500 + step_i)
        assert lib.joker_step(env, act) == 1
        step_i += 1

    _, length_after = get_history_tokens(env, 1)
    assert length_after == 0


def test_observation_tensor_stays_finite_full_match(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=321)
    step_i = 0
    while not lib.joker_is_done(env):
        cur = lib.joker_get_current_player(env)
        obs = get_obs_vec(env, cur)
        assert obs.shape == (320,)
        assert np.all(np.isfinite(obs))
        act = lib.joker_select_heuristic_action(env, 700 + step_i)
        assert lib.joker_step(env, act) == 1
        step_i += 1