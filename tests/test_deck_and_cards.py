import ctypes
import pytest
from conftest import (
    lib, RULES_POPULAR, RULES_MINUS, RULES_CLASSIC,
    MODE_LONG, MODE_SHORT, TEAM_SOLO, PULKA_STRICT,
    PHASE_TRUMP, PHASE_BIDDING, SUIT_SPADES, SUIT_CLUBS, SUIT_DIAMONDS, SUIT_HEARTS,
    RANK_6, RANK_7, RANK_ACE, JK1, JK2,
    cid, popcount, make_custom_deck, get_snap
)


def test_abi_tensor_dimensions():
    assert lib.joker_tensor_size() == 320
    assert lib.joker_oracle_tensor_size() == 160
    assert lib.joker_history_seq_len() == 48
    assert lib.joker_history_token_dim() == 8


@pytest.mark.parametrize("rules,has_black_sixes", [
    (RULES_POPULAR, False),
    (RULES_MINUS, False),
    (RULES_CLASSIC, True),
])
def test_deck_composition_36_vs_38(make_env, rules, has_black_sixes):
    env = make_env(rules=rules, mode=MODE_SHORT, seed=123)
    snap = get_snap(env)
    if rules != RULES_CLASSIC:
        assert snap.phase == PHASE_TRUMP
        lib.joker_step(env, SUIT_SPADES)
        snap = get_snap(env)
    else:
        assert snap.phase == PHASE_BIDDING

    union_hands = snap.hands[0] | snap.hands[1] | snap.hands[2] | snap.hands[3]
    assert popcount(union_hands) == 36

    # Проверяем отсутствие пересечений между руками
    for i in range(4):
        assert popcount(snap.hands[i]) == 9
        for j in range(i + 1, 4):
            assert (snap.hands[i] & snap.hands[j]) == 0

    if not has_black_sixes:
        assert (union_hands & (1 << cid(SUIT_SPADES, RANK_6))) == 0
        assert (union_hands & (1 << cid(SUIT_CLUBS, RANK_6))) == 0


def test_seed_determinism_and_reset256(make_env):
    env1 = make_env(rules=RULES_CLASSIC, mode=MODE_LONG, seed=999)
    env2 = make_env(rules=RULES_CLASSIC, mode=MODE_LONG, seed=999)
    s1, s2 = get_snap(env1), get_snap(env2)
    assert list(s1.hands) == list(s2.hands)
    assert s1.upcard == s2.upcard

    seed256 = (ctypes.c_uint64 * 4)(11, 22, 33, 44)
    lib.joker_reset256(env1, RULES_CLASSIC, MODE_LONG, TEAM_SOLO, PULKA_STRICT, seed256)
    lib.joker_reset256(env2, RULES_CLASSIC, MODE_LONG, TEAM_SOLO, PULKA_STRICT, seed256)
    s1_256, s2_256 = get_snap(env1), get_snap(env2)
    assert list(s1_256.hands) == list(s2_256.hands)


def test_force_deck_order(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_LONG, seed=1)
    deck = make_custom_deck([
        cid(SUIT_DIAMONDS, RANK_ACE),  # P0
        JK1,                           # P1
        cid(SUIT_SPADES, RANK_ACE),    # P2
        cid(SUIT_HEARTS, RANK_6),      # P3
        cid(SUIT_HEARTS, RANK_7),      # upcard
    ])
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 1)

    snap = get_snap(env)
    assert snap.hands[0] == (1 << cid(SUIT_DIAMONDS, RANK_ACE))
    assert snap.hands[1] == (1 << JK1)
    assert snap.hands[2] == (1 << cid(SUIT_SPADES, RANK_ACE))
    assert snap.hands[3] == (1 << cid(SUIT_HEARTS, RANK_6))
    assert snap.upcard == cid(SUIT_HEARTS, RANK_7)
    assert snap.trump_suit == SUIT_HEARTS
