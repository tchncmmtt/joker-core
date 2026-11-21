import pytest
from conftest import (
    lib, RULES_POPULAR, RULES_MINUS, RULES_CLASSIC,
    MODE_LONG, TEAM_SOLO, PULKA_STRICT,
    SUIT_SPADES, SUIT_CLUBS, SUIT_DIAMONDS, SUIT_HEARTS,
    RANK_7, RANK_8, RANK_KING, RANK_ACE,
    cid, make_custom_deck, get_scores,
)


def _play_one_card_deal(make_env, rules, bids_p1_p2_p3_p0):
    """
    1-я сдача (1 карта):
      P0 (дилер): 7♦
      P1 (ходит): A♠  -> выигрывает взятку
      P2:         8♠
      P3:         7♣
      upcard:     K♥ (козырь Червы, ни у кого нет)
    """
    env = make_env(rules=rules, mode=MODE_LONG, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=1)
    deck = make_custom_deck(
        [
            cid(SUIT_DIAMONDS, RANK_7),  # P0
            cid(SUIT_SPADES, RANK_ACE),  # P1 -> берёт взятку
            cid(SUIT_SPADES, RANK_8),    # P2
            cid(SUIT_CLUBS, RANK_7),     # P3
            cid(SUIT_HEARTS, RANK_KING), # upcard -> ♥
        ],
        deck_size=38 if rules == RULES_CLASSIC else 36,
    )
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, rules, MODE_LONG, TEAM_SOLO, PULKA_STRICT, 1)

    for b in bids_p1_p2_p3_p0:
        assert lib.joker_step(env, b) == 1

    # Розыгрыш 1 взятки: P1 -> P2 -> P3 -> P0
    for card in (
            cid(SUIT_SPADES, RANK_ACE),
            cid(SUIT_SPADES, RANK_8),
            cid(SUIT_CLUBS, RANK_7),
            cid(SUIT_DIAMONDS, RANK_7),
    ):
        assert lib.joker_step(env, card) == 1

    return get_scores(env)


@pytest.mark.parametrize("rules,expected_shanga,expected_all_in", [
    (RULES_POPULAR, -200, 100),  # Popular: штанга -200, "все" 1/1 = 100*1
    (RULES_MINUS,   -100, 100),  # Minus:   недобор -100*(1-0), точный 100*1
    (RULES_CLASSIC,    0, 100),  # Classic: штанга 0, точный 50 + 50*1 = 100
])
def test_shanga_and_exact_pass_across_presets(make_env, rules, expected_shanga, expected_all_in):
    # P1=1 (берёт 1), P2=1 (берёт 0 -> штанга/недобор), P3=0 (пас 0/0 -> +50), P0=0 (пас 0/0 -> +50)
    # Сумма заказов = 2 != 1 (висяк не нарушен)
    scores = _play_one_card_deal(make_env, rules, (1, 1, 0, 0))
    assert scores[1] == expected_all_in
    assert scores[2] == expected_shanga
    assert scores[3] == 50
    assert scores[0] == 50


@pytest.mark.parametrize("rules", [RULES_POPULAR, RULES_MINUS, RULES_CLASSIC])
def test_overbid_gives_10_per_trick(make_env, rules):
    # P1 заказал 0, но взял 1 -> перебор (+10); P2=0 (50), P3=0 (50), P0=0 (50) — сумма 0 != 1
    scores = _play_one_card_deal(make_env, rules, (0, 0, 0, 0))
    assert scores[1] == 10
    assert scores[2] == 50
    assert scores[3] == 50
    assert scores[0] == 50