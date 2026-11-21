import ctypes
import numpy as np
from conftest import (
    lib, RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_STRICT,
    PHASE_TRUMP, SUIT_SPADES, SUIT_CLUBS, SUIT_DIAMONDS, SUIT_HEARTS,
    RANK_7, RANK_8, RANK_9, RANK_10, RANK_J, RANK_Q, RANK_K, RANK_A,
    cid, mask_of, popcount, get_snap,
)


def test_clone_determinized_preserves_observer_hand_and_card_counts(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=99)
    # Доходим до розыгрыша
    for step_i in range(10):
        act = lib.joker_select_heuristic_action(env, 300 + step_i)
        assert lib.joker_step(env, act) == 1

    orig = get_snap(env)
    observer = orig.current_player
    det = lib.joker_clone_determinized(env, observer, 123456)
    assert det is not None
    try:
        dsnap = get_snap(det)
        assert dsnap.hands[observer] == orig.hands[observer]
        # Никакие две руки не пересекаются и не содержат уже сыгранных карт
        union = 0
        for p in range(4):
            assert popcount(dsnap.hands[p]) == popcount(orig.hands[p])
            assert (dsnap.hands[p] & orig.played_cards) == 0
            assert (union & dsnap.hands[p]) == 0
            union |= dsnap.hands[p]
    finally:
        lib.joker_destroy(det)


def test_vec_fork_from_observed_env(make_env, make_vec_env):
    """Форк из живого стола (Observed Mode) в пул JokerVectorEnv создаёт валидные детерминизации."""
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=1)
    lib.joker_begin_observed_match(env, RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, 0, 1)

    three = mask_of(cid(SUIT_SPADES, RANK_A), cid(SUIT_SPADES, RANK_K), cid(SUIT_HEARTS, RANK_A))
    assert lib.joker_set_observed_deal(env, three, 255, 255) == 1
    assert get_snap(env).phase == PHASE_TRUMP
    assert lib.joker_step(env, SUIT_SPADES) == 1

    nine = three | mask_of(
        cid(SUIT_SPADES, RANK_7), cid(SUIT_CLUBS, RANK_A), cid(SUIT_CLUBS, RANK_K),
        cid(SUIT_DIAMONDS, RANK_A), cid(SUIT_DIAMONDS, RANK_7), cid(SUIT_HEARTS, RANK_8),
    )
    assert lib.joker_set_observed_remaining_six(env, nine) == 1

    n_worlds = 32
    vec = make_vec_env(num_envs=n_worlds, rules=RULES_POPULAR, mode=MODE_SHORT, threads=2)
    obs = np.zeros((n_worlds, lib.joker_tensor_size()), dtype=np.float32)
    masks = np.zeros(n_worlds, dtype=np.uint64)
    oracle = np.zeros((n_worlds, lib.joker_oracle_tensor_size()), dtype=np.float32)

    lib.joker_vec_fork_from_env(
        vec, env, 1, 7777,
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        oracle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    assert np.all(np.isfinite(obs))
    assert np.all(masks != 0)
    # У каждого мира в Oracle должно быть ровно 36 карт (9*4) и одинаковая рука наблюдателя (слот 0..37)
    for i in range(n_worlds):
        assert int(np.sum(oracle[i, :152])) == 36
        assert np.array_equal(oracle[i, :38], oracle[0, :38])