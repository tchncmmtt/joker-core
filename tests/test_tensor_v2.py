import numpy as np
from conftest import (
    lib, RULES_POPULAR, RULES_CLASSIC,
    MODE_LONG, MODE_SHORT, TEAM_SOLO, TEAM_PAIRS, PULKA_STRICT, PULKA_SOFT,
    SUIT_SPADES, SUIT_CLUBS, SUIT_DIAMONDS, SUIT_HEARTS,
    RANK_6, RANK_7, RANK_8, RANK_9, RANK_10, RANK_J, RANK_Q, RANK_K, RANK_A,
    JK1, ACT_JOKER_LEAD_HIGH_BASE,
    cid, make_custom_deck, get_obs_vec, get_snap,
)


def test_tensor_v2_horizon_long_vs_short(make_env):
    env_long = make_env(rules=RULES_POPULAR, mode=MODE_LONG, seed=10)
    obs_l = get_obs_vec(env_long, 1)
    assert obs_l.shape == (320,)
    assert np.isclose(obs_l[3], 1.0 / 9.0)    # cardsInDeal = 1
    assert np.isclose(obs_l[11], 36.0 / 36.0) # remainingCardsInPulka = 36
    assert np.isclose(obs_l[14], 2.0 / 9.0)   # nextDealCards = 2 (подъём)
    assert np.isclose(obs_l[16], 3.0 / 3.0)   # 3 пульки после текущей
    assert np.isclose(obs_l[17], 144.0 / 144.0)

    env_short = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, seed=10)
    obs_s = get_obs_vec(env_short, 1)
    assert np.isclose(obs_s[3], 1.0)           # cardsInDeal = 9
    assert obs_s[13] == 1.0                    # isNinesPulka
    assert np.isclose(obs_s[16], 1.0 / 3.0)    # 1 пулька после текущей
    assert np.isclose(obs_s[17], 72.0 / 144.0) # 0.5 от полного матча


def test_tensor_v2_rank_ceiling_master_cards_and_trick_mask(make_env):
    env = make_env(rules=RULES_CLASSIC, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=7)
    p0 = [
        cid(SUIT_SPADES, RANK_A), cid(SUIT_SPADES, RANK_K),
        cid(SUIT_HEARTS, RANK_A), cid(SUIT_HEARTS, RANK_K),
        cid(SUIT_DIAMONDS, RANK_A), cid(SUIT_DIAMONDS, RANK_K),
        cid(SUIT_CLUBS, RANK_A), cid(SUIT_CLUBS, RANK_K), cid(SUIT_CLUBS, RANK_Q),
    ]
    p1 = [
        JK1, cid(SUIT_SPADES, RANK_Q), cid(SUIT_SPADES, RANK_J),
        cid(SUIT_HEARTS, RANK_Q), cid(SUIT_HEARTS, RANK_J),
        cid(SUIT_DIAMONDS, RANK_Q), cid(SUIT_DIAMONDS, RANK_J),
        cid(SUIT_CLUBS, RANK_J), cid(SUIT_CLUBS, RANK_10),
    ]
    p2 = [
        cid(SUIT_SPADES, RANK_10), cid(SUIT_SPADES, RANK_9), cid(SUIT_SPADES, RANK_8),
        cid(SUIT_SPADES, RANK_7), cid(SUIT_SPADES, RANK_6),
        cid(SUIT_DIAMONDS, RANK_10), cid(SUIT_DIAMONDS, RANK_9),
        cid(SUIT_CLUBS, RANK_9), cid(SUIT_CLUBS, RANK_8),
    ]
    p3 = [
        cid(SUIT_HEARTS, RANK_9), cid(SUIT_HEARTS, RANK_8), cid(SUIT_HEARTS, RANK_7), cid(SUIT_HEARTS, RANK_6),
        cid(SUIT_DIAMONDS, RANK_8), cid(SUIT_DIAMONDS, RANK_7), cid(SUIT_DIAMONDS, RANK_6),
        cid(SUIT_CLUBS, RANK_7), cid(SUIT_CLUBS, RANK_6),
    ]
    upcard = cid(SUIT_HEARTS, RANK_10)

    lib.joker_force_deck_order(env, make_custom_deck(p0 + p1 + p2 + p3 + [upcard], deck_size=38))
    lib.joker_reset(env, RULES_CLASSIC, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, 7)

    for b in (2, 2, 2, 2):
        assert lib.joker_step(env, b) == 1

    # P1 ходит JK1 LeadHighSuit(Spades), P2 даёт 10♠ -> у P2 потолок пик <= 9♠ (rank=3 -> 4/9)
    assert lib.joker_step(env, ACT_JOKER_LEAD_HIGH_BASE + SUIT_SPADES) == 1
    assert lib.joker_step(env, cid(SUIT_SPADES, RANK_10)) == 1

    # Проверяем наблюдение от лица P0 прямо в середине взятки (2 карты на столе)
    obs = get_obs_vec(env, 0)
    # Единая маска текущей взятки (234..271): JK1 и 10♠
    assert obs[234 + JK1] == 1.0
    assert obs[234 + cid(SUIT_SPADES, RANK_10)] == 1.0
    assert int(np.sum(obs[234:272])) == 2

    # Потолок пик для P2 (rel=2 -> индекс 284 + (2-1)*4 + 0 = 288): максимум 9♠ (rank=3 -> 4/9)
    assert np.isclose(obs[288], 4.0 / 9.0)

    # У P0 на руках A♠ и K♠ — оба мастера (2/4 = 0.5)
    assert np.isclose(obs[310 + SUIT_SPADES], 2.0 / 4.0)
    # Вся взятка держится Джокером (obs[98] == 1.0)
    assert obs[98] == 1.0
    # Наблюдение без NaN/Inf
    assert np.all(np.isfinite(obs))


def test_tensor_v2_pairs_and_counterfactual_swings(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_PAIRS, pulka=PULKA_SOFT, seed=42)
    lib.joker_step(env, SUIT_HEARTS)  # козырь
    obs = get_obs_vec(env, 0)
    assert obs.shape == (320,)
    assert np.all(np.isfinite(obs))
    # На старте счёты равны -> относительные отрывы строго 0.0
    assert obs[18] == 0.0 and obs[19] == 0.0 and obs[20] == 0.0
    # В Pairs+Soft все 4 игрока чистые и на всех пасах -> если P0 сорвётся, он потеряет 0?
    # В Soft P0 и так заблокирован выходом P3 (over=3), но если P3 сорвётся -> P0 разблокирует +450!
    assert obs[38 + 3] > 0.0  # enemyOverFailureNetSwing > 0