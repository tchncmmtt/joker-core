import pytest
from conftest import (
    lib, RULES_POPULAR, MODE_LONG, MODE_SHORT, TEAM_SOLO, TEAM_PAIRS,
    PULKA_STRICT, PULKA_SOFT,
    get_snap, get_scores, get_team_scores,
)


@pytest.mark.parametrize("mode,expected_deals", [
    (MODE_LONG, 24),
    (MODE_SHORT, 8),
])
def test_match_mode_deal_counts(make_env, mode, expected_deals):
    env = make_env(rules=RULES_POPULAR, mode=mode, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=55)
    seen_deals = set()
    step_i = 0
    while not lib.joker_is_done(env):
        seen_deals.add(lib.joker_get_deal_number(env))
        act = lib.joker_select_heuristic_action(env, 1000 + step_i)
        assert lib.joker_step(env, act) == 1
        step_i += 1
    assert len(seen_deals) == expected_deals


@pytest.mark.parametrize("pulka", [PULKA_STRICT, PULKA_SOFT])
def test_pairs_team_scores_equal_sum_of_partners(make_env, pulka):
    """В парном режиме team_scores == [P0 + P2, P1 + P3] на всём протяжении матча."""
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_PAIRS, pulka=pulka, seed=77)
    step_i = 0
    while not lib.joker_is_done(env):
        act = lib.joker_select_heuristic_action(env, 2000 + step_i)
        assert lib.joker_step(env, act) == 1
        step_i += 1
        if step_i % 20 == 0 or lib.joker_is_done(env):
            sc = get_scores(env)
            ts = get_team_scores(env)
            assert ts[0] == sc[0] + sc[2]
            assert ts[1] == sc[1] + sc[3]