from conftest import (
    lib, RULES_POPULAR, MODE_LONG,
    SUIT_SPADES, SUIT_DIAMONDS, SUIT_HEARTS,
    RANK_6, RANK_7, RANK_KING, RANK_ACE, JK1, JK2,
    ACT_JOKER_PLAY_WIN, ACT_JOKER_PLAY_LOW,
    ACT_JOKER_LEAD_HIGH_BASE, ACT_JOKER_LEAD_LOW_BASE,
    cid, make_custom_deck, get_scores
)


def test_joker_lead_high_suit_trumped_by_void_opponent(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_LONG, seed=1)
    deck = make_custom_deck([
        cid(SUIT_DIAMONDS, RANK_ACE),  # P0 (дилер)
        JK1,                           # P1 (ходит первым)
        cid(SUIT_SPADES, RANK_ACE),    # P2 (есть пика -> обязан дать A♠)
        cid(SUIT_HEARTS, RANK_6),      # P3 (нет пик -> обязан дать козырь 6♥ и забрать!)
        cid(SUIT_HEARTS, RANK_KING),   # upcard -> козырь Червы
    ])
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, RULES_POPULAR, MODE_LONG, 0, 0, 1)

    for b in (0, 0, 1, 1):
        lib.joker_step(env, b)

    assert lib.joker_step(env, ACT_JOKER_LEAD_HIGH_BASE + SUIT_SPADES) == 1
    assert lib.joker_step(env, cid(SUIT_SPADES, RANK_ACE)) == 1
    assert lib.joker_step(env, cid(SUIT_HEARTS, RANK_6)) == 1
    assert lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_ACE)) == 1

    scores = get_scores(env)
    assert scores[3] == 100


def test_joker_lead_low_suit_loses_to_ordered_suit(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_LONG, seed=2)
    deck = make_custom_deck([
        cid(SUIT_DIAMONDS, RANK_6),   # P0 (дилер)
        JK1,                          # P1 (LeadLow Пики = 42 + 0)
        cid(SUIT_SPADES, RANK_7),     # P2 (7♠ забирает у LeadLow!)
        cid(SUIT_DIAMONDS, RANK_7),   # P3
        cid(SUIT_HEARTS, RANK_KING),  # upcard -> Червы
    ])
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, RULES_POPULAR, MODE_LONG, 0, 0, 2)

    for b in (0, 1, 0, 1):
        assert lib.joker_step(env, b) == 1

    assert lib.joker_step(env, ACT_JOKER_LEAD_LOW_BASE + SUIT_SPADES) == 1
    assert lib.joker_step(env, cid(SUIT_SPADES, RANK_7)) == 1
    assert lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_7)) == 1
    assert lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_6)) == 1

    scores = get_scores(env)
    assert scores[2] == 100


def test_two_jokers_play_win_second_takes_trick(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_LONG, seed=3)
    deck = make_custom_deck([
        cid(SUIT_DIAMONDS, RANK_6),   # P0 (дилер)
        cid(SUIT_SPADES, RANK_7),     # P1
        JK1,                          # P2 (PlayWin = 36)
        JK2,                          # P3 (PlayWin = 36 -> побеждает!)
        cid(SUIT_HEARTS, RANK_KING),  # upcard
    ])
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, RULES_POPULAR, MODE_LONG, 0, 0, 3)

    for b in (0, 0, 1, 1):
        lib.joker_step(env, b)

    assert lib.joker_step(env, cid(SUIT_SPADES, RANK_7)) == 1
    assert lib.joker_step(env, ACT_JOKER_PLAY_WIN) == 1
    assert lib.joker_step(env, ACT_JOKER_PLAY_WIN) == 1
    assert lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_6)) == 1

    scores = get_scores(env)
    assert scores[3] == 100 and scores[2] == 50


def test_joker_play_low_always_loses(make_env):
    env = make_env(rules=RULES_POPULAR, mode=MODE_LONG, seed=4)
    deck = make_custom_deck([
        cid(SUIT_DIAMONDS, RANK_6),   # P0 (дилер)
        cid(SUIT_SPADES, RANK_7),     # P1 (забирает семёркой!)
        JK1,                          # P2 (PlayLow = 37)
        cid(SUIT_DIAMONDS, RANK_7),   # P3
        cid(SUIT_HEARTS, RANK_KING),  # upcard
    ])
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, RULES_POPULAR, MODE_LONG, 0, 0, 4)

    for b in (1, 0, 0, 1):
        lib.joker_step(env, b)

    assert lib.joker_step(env, cid(SUIT_SPADES, RANK_7)) == 1
    assert lib.joker_step(env, ACT_JOKER_PLAY_LOW) == 1
    assert lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_7)) == 1
    assert lib.joker_step(env, cid(SUIT_DIAMONDS, RANK_6)) == 1

    scores = get_scores(env)
    assert scores[1] == 100 and scores[2] == 50