import pytest
from conftest import (
    lib, RULES_POPULAR, RULES_MINUS, RULES_CLASSIC,
    MODE_LONG, MODE_SHORT, PHASE_TRUMP, PHASE_BIDDING,
    SUIT_SPADES, SUIT_CLUBS, SUIT_DIAMONDS, SUIT_HEARTS, SUIT_NONE,
    RANK_7, JK1,
    cid, legal_actions, make_custom_deck, get_snap, get_obs_vec
)


def test_trump_choosing_on_nines_popular_vs_classic(make_env):
    # В Popular и Minus на девятках (Short) козырь выбирает игрок слева от дилера (P1 при дилере P0)
    for rules in (RULES_POPULAR, RULES_MINUS):
        env = make_env(rules=rules, mode=MODE_SHORT, seed=42)
        snap = get_snap(env)
        assert snap.phase == PHASE_TRUMP
        assert snap.dealer == 0
        assert snap.current_player == 1
        assert legal_actions(lib.joker_get_legal_actions_mask(env)) == [0, 1, 2, 3, 4]

        lib.joker_step(env, SUIT_NONE)
        snap = get_snap(env)
        assert snap.phase == PHASE_BIDDING
        assert snap.trump_suit == SUIT_NONE
        assert snap.current_player == 1

    # В Classic на девятках козырь вскрывается из колоды сразу (фазы ChoosingTrump нет)
    env_classic = make_env(rules=RULES_CLASSIC, mode=MODE_SHORT, seed=42)
    snap_c = get_snap(env_classic)
    assert snap_c.phase == PHASE_BIDDING
    assert snap_c.current_player == 1


def test_upcard_joker_sets_no_trump(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_LONG, seed=1)
    deck = make_custom_deck([
        cid(SUIT_SPADES, RANK_7),
        cid(SUIT_CLUBS, RANK_7),
        cid(SUIT_DIAMONDS, RANK_7),
        cid(SUIT_HEARTS, RANK_7),
        JK1,  # вскрылся Джокер -> бескозырка
    ])
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, RULES_POPULAR, MODE_LONG, 0, 0, 1)

    snap = get_snap(env)
    assert snap.upcard == JK1
    assert snap.trump_suit == SUIT_NONE


def test_visyak_dealer_restriction_and_tensor(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_LONG, seed=10)
    # 1-я сдача (1 карта), дилер P0, торгуют P1 -> P2 -> P3 -> P0
    assert lib.joker_get_current_player(env) == 1
    lib.joker_step(env, 0)
    lib.joker_step(env, 0)
    lib.joker_step(env, 0)

    assert lib.joker_get_current_player(env) == 0
    acts = legal_actions(lib.joker_get_legal_actions_mask(env))
    assert acts == [0], f"Висяк 1 должен быть запрещён, доступно: {acts}"

    obs = get_obs_vec(env, 0)
    assert obs[71] == 1.0 and obs[70] == 0.0
