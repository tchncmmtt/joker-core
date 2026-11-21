import ctypes
import numpy as np
from conftest import (
    lib, JokerRewardConfig,
    RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_STRICT,
)


def test_vec_env_step_batch_and_auto_reset(make_vec_env):
    n = 16
    vec = make_vec_env(num_envs=n, rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=11, threads=2)
    lib.joker_vec_set_auto_reset(vec, 1)

    cfg = JokerRewardConfig(
        score_scale=0.01,
        relative_weight=0.5,
        match_win_bonus=1.0,
        deal_exact_bonus=0.1,
        trick_win_shaping=0.02,
    )
    lib.joker_vec_set_reward_config(vec, ctypes.byref(cfg))

    obs = np.zeros((n, lib.joker_tensor_size()), dtype=np.float32)
    masks = np.zeros(n, dtype=np.uint64)
    players = np.zeros(n, dtype=np.uint8)
    rewards = np.zeros((n, 4), dtype=np.float32)
    deal_dones = np.zeros(n, dtype=np.uint8)
    match_dones = np.zeros(n, dtype=np.uint8)
    scores = np.zeros((n, 4), dtype=np.int32)
    oracle = np.zeros((n, lib.joker_oracle_tensor_size()), dtype=np.float32)
    hist = np.zeros((n, lib.joker_history_seq_len(), lib.joker_history_token_dim()), dtype=np.int16)
    hist_len = np.zeros(n, dtype=np.uint8)

    lib.joker_vec_get_observations(
        vec,
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        players.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        oracle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        hist_len.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )

    actions = np.zeros(n, dtype=np.uint8)
    total_matches = 0

    for _ in range(400):
        lsb = masks & (~masks + np.uint64(1))
        actions[:] = np.log2(np.maximum(lsb, np.uint64(1))).astype(np.uint8)
        lib.joker_vec_step(
            vec,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            players.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            deal_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            match_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            oracle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            hist_len.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        total_matches += int(np.sum(match_dones))
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))

    assert total_matches >= n


def test_vec_env_advance_to_hero_and_step_arena(make_vec_env):
    n = 16
    vec = make_vec_env(num_envs=n, rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=22, threads=2)
    hero_seats = np.array([i % 4 for i in range(n)], dtype=np.uint8)
    obs = np.zeros((n, lib.joker_tensor_size()), dtype=np.float32)
    masks = np.zeros(n, dtype=np.uint64)
    hero_rewards = np.zeros(n, dtype=np.float32)
    all_rewards = np.zeros((n, 4), dtype=np.float32)
    deal_dones = np.zeros(n, dtype=np.uint8)
    match_dones = np.zeros(n, dtype=np.uint8)
    scores = np.zeros((n, 4), dtype=np.int32)

    lib.joker_vec_advance_to_hero(
        vec,
        hero_seats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        None, None, None,
    )
    assert np.all(masks != 0)

    actions = np.zeros(n, dtype=np.uint8)
    matches = 0
    for _ in range(120):
        lsb = masks & (~masks + np.uint64(1))
        actions[:] = np.log2(np.maximum(lsb, np.uint64(1))).astype(np.uint8)
        lib.joker_vec_step_arena(
            vec,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            hero_seats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            hero_rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            all_rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            deal_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            match_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            None, None, None,
        )
        matches += int(np.sum(match_dones))
        assert np.all(np.isfinite(hero_rewards))

    assert matches >= n