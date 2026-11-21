import ctypes
import time
import numpy as np
import pytest
from conftest import (
    lib, JokerPlayerStats,
    RULES_POPULAR, RULES_MINUS, RULES_CLASSIC,
    MODE_LONG, MODE_SHORT, TEAM_SOLO, PULKA_STRICT,
    PHASE_TRUMP, PHASE_PLAY, SUIT_NONE,
    ACT_JOKER_PLAY_WIN, ACT_JOKER_PLAY_LOW,
    ACT_JOKER_LEAD_HIGH_BASE, ACT_JOKER_LEAD_LOW_BASE,
    legal_actions, get_snap, get_scores,
)


def _calc_deal_score(rules: int, bid: int, tricks: int, n_cards: int) -> int:
    if bid == 0 and tricks == 0:
        return 50
    if tricks == bid:
        if rules == RULES_MINUS:
            return 100 * tricks
        if rules == RULES_POPULAR and bid == n_cards:
            return 100 * tricks
        return 50 + 50 * tricks
    if tricks > bid:
        return 10 * tricks
    if rules == RULES_MINUS:
        return -100 * (bid - tricks)
    if tricks == 0:
        return -200 if rules == RULES_POPULAR else 0
    return 10 * tricks


@pytest.mark.parametrize("rules", [RULES_POPULAR, RULES_MINUS, RULES_CLASSIC])
def test_heuristic_beats_random_over_90_percent(make_env, rules):
    """Эвристика (P0) против 3 случайных игроков (P1..P3) должна выигрывать > 90% партий во всех 3 пресетах."""
    n_matches = 60
    wins = 0.0
    rng = np.random.default_rng(1234 + rules)

    for m in range(n_matches):
        env = make_env(rules=rules, mode=MODE_LONG, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=1000 + m)
        step_i = 0
        while not lib.joker_is_done(env):
            cur = lib.joker_get_current_player(env)
            if cur == 0:
                act = lib.joker_select_heuristic_action(env, 9001 + m * 1000 + step_i)
            else:
                acts = legal_actions(lib.joker_get_legal_actions_mask(env))
                act = int(acts[int(rng.integers(0, len(acts)))])
            assert lib.joker_step(env, act) == 1
            step_i += 1

        sc = get_scores(env)
        max_sc = max(sc)
        if sc[0] == max_sc:
            wins += 1.0 / sc.count(max_sc)

    winrate = wins / n_matches
    assert winrate > 0.90, f"rules={rules}: winrate={winrate:.2%} (ожидалось > 90%)"


def test_player_stats_exact_match_with_python_reference(make_env, make_vec_env):
    """Пошаговая сверка всех счётчиков JokerPlayerStats против эталонного подсчёта на Python."""
    seed = 777
    vec = make_vec_env(num_envs=1, rules=RULES_POPULAR, mode=MODE_LONG, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=seed, threads=1)
    lib.joker_vec_set_auto_reset(vec, 0)
    lib.joker_vec_set_stats_enabled(vec, 1)
    lib.joker_vec_reset_stats(vec)

    seeds_arr = (ctypes.c_uint64 * 1)(seed)
    lib.joker_vec_reset_with_seeds(vec, RULES_POPULAR, MODE_LONG, TEAM_SOLO, PULKA_STRICT, seeds_arr)

    ref_env = make_env(rules=RULES_POPULAR, mode=MODE_LONG, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=seed)

    # Эталонные счётчики на Python
    ref = [dict(
        matches_played=0, place_counts=[0, 0, 0, 0], place_points=0.0,
        total_match_score_sum=0, total_match_score_sq_sum=0.0,
        max_match_score=-10**9, min_match_score=10**9,
        deals_played=0, max_deal_score=-10**9, min_deal_score=10**9,
        exact_bids=0, overbids=0, underbids=0, shangas=0,
        large_bids_count=0, large_bids_exact=0,
        deals_1_8_count=0, bids_sum_1_8=0, tricks_sum_1_8=0,
        deals_9_count=0, bids_sum_9=0, tricks_sum_9=0,
        nines_trump_choices=0, nines_no_trump_choices=0,
        joker_lead_high=0, joker_lead_low=0, joker_play_win=0, joker_play_low=0,
        pulkas_played=0, pulkas_clean=0, pulkas_bonus_received=0, pulkas_penalty_received=0,
        matches_multi_bonus=0,
    ) for _ in range(4)]

    pulka_end_deals = {7, 11, 19, 23}
    match_bonuses = [0, 0, 0, 0]

    act_buf = np.zeros(1, dtype=np.uint8)
    match_done_buf = np.zeros(1, dtype=np.uint8)
    step_i = 0

    while not lib.joker_is_done(ref_env):
        sb = get_snap(ref_env)
        act = lib.joker_select_heuristic_action(ref_env, 5000 + step_i)
        step_i += 1

        # Оба движка делают строго одинаковый шаг
        assert lib.joker_step(ref_env, act) == 1
        act_buf[0] = act
        lib.joker_vec_step(
            vec, act_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            None, None, None, None, None,
            match_done_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            None, None, None, None,
        )

        sa = get_snap(ref_env)
        actor = sb.current_player

        if sb.phase == PHASE_TRUMP:
            ref[actor]["nines_trump_choices"] += 1
            if act == SUIT_NONE:
                ref[actor]["nines_no_trump_choices"] += 1
        elif sb.phase == PHASE_PLAY and act >= 36:
            if act == ACT_JOKER_PLAY_WIN:
                ref[actor]["joker_play_win"] += 1
            elif act == ACT_JOKER_PLAY_LOW:
                ref[actor]["joker_play_low"] += 1
            elif ACT_JOKER_LEAD_HIGH_BASE <= act < ACT_JOKER_LEAD_HIGH_BASE + 4:
                ref[actor]["joker_lead_high"] += 1
            elif ACT_JOKER_LEAD_LOW_BASE <= act < ACT_JOKER_LEAD_LOW_BASE + 4:
                ref[actor]["joker_lead_low"] += 1

        match_done = bool(lib.joker_is_done(ref_env))
        deal_done = match_done or (sa.deal_number != sb.deal_number)

        if deal_done:
            tricks_final = list(sb.tricks_taken)
            if sb.phase == PHASE_PLAY and sb.cards_in_trick == 3:
                tricks_final[sa.last_trick_winner] += 1

            for p in range(4):
                b, t = sb.bids[p], tricks_final[p]
                dsc = _calc_deal_score(RULES_POPULAR, b, t, sb.cards_in_deal)
                r = ref[p]
                r["deals_played"] += 1
                r["max_deal_score"] = max(r["max_deal_score"], dsc)
                r["min_deal_score"] = min(r["min_deal_score"], dsc)
                if t == b:
                    r["exact_bids"] += 1
                elif t > b:
                    r["overbids"] += 1
                else:
                    r["underbids"] += 1
                    if b > 0 and t == 0:
                        r["shangas"] += 1
                if b >= 4:
                    r["large_bids_count"] += 1
                    if t == b:
                        r["large_bids_exact"] += 1
                if sb.cards_in_deal == 9:
                    r["deals_9_count"] += 1
                    r["bids_sum_9"] += b
                    r["tricks_sum_9"] += t
                else:
                    r["deals_1_8_count"] += 1
                    r["bids_sum_1_8"] += b
                    r["tricks_sum_1_8"] += t

                if sb.deal_number in pulka_end_deals:
                    r["pulkas_played"] += 1
                    if sb.pulka_clean[p] and b == t:
                        r["pulkas_clean"] += 1
                    pdelta = (sa.total_scores[p] - sb.total_scores[p]) - dsc
                    if pdelta > 0:
                        r["pulkas_bonus_received"] += 1
                        match_bonuses[p] += 1
                    elif pdelta < 0:
                        r["pulkas_penalty_received"] += 1

        if match_done:
            sc = list(sa.total_scores)
            for p in range(4):
                better = sum(1 for q in range(4) if sc[q] > sc[p])
                eq = sum(1 for q in range(4) if sc[q] == sc[p])
                r = ref[p]
                r["matches_played"] += 1
                r["place_counts"][better] += 1
                if better == 0:
                    r["place_points"] += 1.0 / eq
                r["total_match_score_sum"] += sc[p]
                r["total_match_score_sq_sum"] += float(sc[p] * sc[p])
                r["max_match_score"] = max(r["max_match_score"], sc[p])
                r["min_match_score"] = min(r["min_match_score"], sc[p])
                if match_bonuses[p] >= 2:
                    r["matches_multi_bonus"] += 1

    c_stats = (JokerPlayerStats * 4)()
    lib.joker_vec_get_stats(vec, c_stats)

    for p in range(4):
        s, r = c_stats[p], ref[p]
        for field in [
            "matches_played", "total_match_score_sum", "max_match_score", "min_match_score",
            "deals_played", "max_deal_score", "min_deal_score",
            "exact_bids", "overbids", "underbids", "shangas",
            "large_bids_count", "large_bids_exact",
            "deals_1_8_count", "bids_sum_1_8", "tricks_sum_1_8",
            "deals_9_count", "bids_sum_9", "tricks_sum_9",
            "nines_trump_choices", "nines_no_trump_choices",
            "joker_lead_high", "joker_lead_low", "joker_play_win", "joker_play_low",
            "pulkas_played", "pulkas_clean", "pulkas_bonus_received", "pulkas_penalty_received",
            "matches_multi_bonus",
        ]:
            assert getattr(s, field) == r[field], f"P{p}.{field}: C={getattr(s, field)} != Py={r[field]}"
        assert list(s.place_counts) == r["place_counts"]
        assert abs(s.place_points - r["place_points"]) < 1e-9
        assert abs(s.total_match_score_sq_sum - r["total_match_score_sq_sum"]) < 1e-6


def test_duplicate_matches_and_auto_reset_off(make_vec_env):
    """Зеркальные раздачи через reset_with_seeds дают идентичные состояния; auto_reset=0 останавливает завершённые среды."""
    n = 8
    seeds = (ctypes.c_uint64 * n)(*[100 + i for i in range(n)])
    v1 = make_vec_env(num_envs=n, rules=RULES_POPULAR, mode=MODE_SHORT, threads=2)
    v2 = make_vec_env(num_envs=n, rules=RULES_POPULAR, mode=MODE_SHORT, threads=2)

    lib.joker_vec_reset_with_seeds(v1, RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, seeds)
    lib.joker_vec_reset_with_seeds(v2, RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, seeds)

    obs1 = np.zeros((n, lib.joker_tensor_size()), dtype=np.float32)
    obs2 = np.zeros((n, lib.joker_tensor_size()), dtype=np.float32)
    m1 = np.zeros(n, dtype=np.uint64)
    m2 = np.zeros(n, dtype=np.uint64)

    lib.joker_vec_get_observations(v1, obs1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), None, None, None, None)
    lib.joker_vec_get_observations(v2, obs2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m2.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), None, None, None, None)

    assert np.array_equal(obs1, obs2)
    assert np.array_equal(m1, m2)

    # Проверяем auto_reset=0
    lib.joker_vec_set_auto_reset(v1, 0)
    actions = np.zeros(n, dtype=np.uint8)
    dones = np.zeros(n, dtype=np.uint8)
    total_finished = 0

    for _ in range(400):
        for i in range(n):
            actions[i] = int(int(m1[i]).bit_length() - 1) if m1[i] != 0 else 0
        lib.joker_vec_step(
            v1, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            None, m1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            None, None, None, dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            None, None, None, None,
        )
        total_finished += int(np.sum(dones))
        if np.all(m1 == 0):
            break

    assert total_finished == n
    assert np.all(m1 == 0)


@pytest.mark.perf
def test_vector_env_throughput_benchmark(make_vec_env):
    """Бенчмарк пропускной способности JokerVectorEnv (OpenMP + C++ эвристика)."""
    n = 512
    vec = make_vec_env(num_envs=n, rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=42, threads=0)
    hero_seats = np.zeros(n, dtype=np.uint8)
    masks = np.zeros(n, dtype=np.uint64)
    actions = np.zeros(n, dtype=np.uint8)
    match_dones = np.zeros(n, dtype=np.uint8)

    lib.joker_vec_advance_to_hero(
        vec,
        hero_seats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        None,
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        None, None, None,
    )

    matches = 0
    t0 = time.perf_counter()
    # В StepArena 3 из 4 игроков ходят C++ эвристикой в OpenMP
    for _ in range(300):
        # Быстрый векторизованный выбор минимального легального бита в NumPy
        lsb = masks & (~masks + np.uint64(1))
        actions[:] = np.log2(np.maximum(lsb, np.uint64(1))).astype(np.uint8)
        lib.joker_vec_step_arena(
            vec,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            hero_seats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            None,
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            None, None, None,
            match_dones.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            None, None, None, None,
        )
        matches += int(np.sum(match_dones))

    dt = time.perf_counter() - t0
    mps = matches / dt
    print(f"\n[PERF] JokerVectorEnv (Short, 3x HeuristicBot + OpenMP): {matches} matches in {dt:.3f}s = {mps:,.0f} matches/sec")
    assert mps >= 10000.0, f"Производительность ниже 10 000 партий/сек: {mps:,.0f}"