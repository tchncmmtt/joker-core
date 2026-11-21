import ctypes
import numpy as np
import pytest
from conftest import (
    lib, RULES_POPULAR, RULES_CLASSIC,
    MODE_SHORT, TEAM_SOLO, TEAM_PAIRS, PULKA_STRICT, PULKA_SOFT,
    PHASE_TRUMP, PHASE_BIDDING, PHASE_PLAY,
    SUIT_SPADES, SUIT_CLUBS, SUIT_DIAMONDS, SUIT_HEARTS, SUIT_NONE,
    RANK_6, RANK_7, RANK_8, RANK_9, RANK_10, RANK_J, RANK_Q, RANK_K, RANK_A,
    JK1, JK2, INVALID_CARD,
    ACT_JOKER_LEAD_HIGH_BASE, ACT_JOKER_PLAY_WIN,
    JOKER_ERR_OK, JOKER_ERR_WAITING_FOR_DEAL_INPUT,
    JOKER_ERR_FORBIDDEN_DEALER_BID, JOKER_ERR_CARD_ALREADY_PLAYED,
    JOKER_ERR_CARD_IN_OTHER_KNOWN_HAND, JOKER_ERR_VIOLATES_KNOWN_VOID_OR_BOUND,
    cid, mask_of, legal_actions, popcount, make_custom_deck,
    get_snap, get_scores, get_history_tokens,
)


def _calc_deal_score_popular(bid: int, tricks: int, n_cards: int) -> int:
    if bid == 0 and tricks == 0:
        return 50
    if tricks == bid:
        return 100 * tricks if bid == n_cards else 50 + 50 * tricks
    if tricks > bid:
        return 10 * tricks
    return -200 if tricks == 0 else 10 * tricks


@pytest.mark.parametrize("team_mode", [TEAM_SOLO, TEAM_PAIRS])
def test_soft_pulka_over_is_p_plus_3_mod_4(make_env, team_mode):
    """Проверка правила Soft из RULES.md п. 1.11: сосед 'над вами' over = (P + 3) % 4."""
    asymmetric_verified = False

    for seed in range(1, 30):
        env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=team_mode, pulka=PULKA_SOFT, seed=seed)
        pulka_clean = [True] * 4
        pulka_all_passes = [True] * 4
        max_excl = [0] * 4
        max_non_pass_excl = [0] * 4
        expected_scores = [0] * 4

        # Отыгрываем ровно 1-ю пульку (4 сдачи по 9 карт)
        while get_snap(env).deal_number < 4 and not lib.joker_is_done(env):
            snap_before = get_snap(env)
            prev_deal = snap_before.deal_number
            bids_before = list(snap_before.bids)
            tricks_before = list(snap_before.tricks_taken)
            was_last_card = (snap_before.phase == PHASE_PLAY and snap_before.cards_in_trick == 3)

            act = lib.joker_select_heuristic_action(env, seed * 1000 + prev_deal * 37 + snap_before.current_player)
            assert lib.joker_step(env, act) == 1

            snap_after = get_snap(env)
            if snap_after.deal_number != prev_deal:
                tricks_final = list(tricks_before)
                if was_last_card:
                    tricks_final[snap_after.last_trick_winner] += 1

                for p in range(4):
                    b, t = bids_before[p], tricks_final[p]
                    sc = _calc_deal_score_popular(b, t, 9)
                    expected_scores[p] += sc
                    if b != t:
                        pulka_clean[p] = False
                    if b != t or b != 0:
                        pulka_all_passes[p] = False
                    if prev_deal < 3:  # кроме последней сдачи пульки
                        max_excl[p] = max(max_excl[p], sc)
                        if not (b == 0 and t == 0):
                            max_non_pass_excl[p] = max(max_non_pass_excl[p], sc)

                if prev_deal == 3:
                    # Эталонный Soft: over = (p + 3) % 4
                    scores_prev3 = list(expected_scores)
                    scores_wrong_plus1 = list(expected_scores)
                    for p in range(4):
                        over = (p + 3) % 4
                        if pulka_clean[p]:
                            if not pulka_clean[over]:
                                scores_prev3[p] += 450 if pulka_all_passes[p] else max_excl[p]
                        elif pulka_clean[over]:
                            scores_prev3[p] -= max_non_pass_excl[p]

                        wrong_over = (p + 1) % 4
                        if pulka_clean[p]:
                            if not pulka_clean[wrong_over]:
                                scores_wrong_plus1[p] += 450 if pulka_all_passes[p] else max_excl[p]
                        elif pulka_clean[wrong_over]:
                            scores_wrong_plus1[p] -= max_non_pass_excl[p]

                    actual = get_scores(env)
                    assert actual == scores_prev3
                    if scores_prev3 != scores_wrong_plus1:
                        asymmetric_verified = True
                    break

    assert asymmetric_verified, "Не попалась асимметричная пулька для отличия (P+3)%4 от (P+1)%4"


def test_impossible_cards_lead_high_suit_and_determinize(make_env):
    """После LeadHighSuit карты строго старше сыгранной исключаются и никогда не попадают в Determinize."""
    env = make_env(rules=RULES_CLASSIC, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=7)

    # 38-карточная колода (Classic): сразу по 9 карт + 37-я карта upcard (Червы -> козырь ♥)
    p0_cards = [
        cid(SUIT_SPADES, RANK_A), cid(SUIT_SPADES, RANK_K),
        cid(SUIT_HEARTS, RANK_A), cid(SUIT_HEARTS, RANK_K),
        cid(SUIT_DIAMONDS, RANK_A), cid(SUIT_DIAMONDS, RANK_K),
        cid(SUIT_CLUBS, RANK_A), cid(SUIT_CLUBS, RANK_K),
        cid(SUIT_CLUBS, RANK_Q),
    ]
    p1_cards = [
        JK1,
        cid(SUIT_SPADES, RANK_Q), cid(SUIT_SPADES, RANK_J),
        cid(SUIT_HEARTS, RANK_Q), cid(SUIT_HEARTS, RANK_J),
        cid(SUIT_DIAMONDS, RANK_Q), cid(SUIT_DIAMONDS, RANK_J),
        cid(SUIT_CLUBS, RANK_J), cid(SUIT_CLUBS, RANK_10),
    ]
    # P2: есть пики 10♠ и 7♠ -> на LeadHighSuit(♠) обязан дать 10♠ => нет J♠, Q♠, K♠, A♠
    p2_cards = [
        cid(SUIT_SPADES, RANK_10), cid(SUIT_SPADES, RANK_7),
        cid(SUIT_DIAMONDS, RANK_10), cid(SUIT_DIAMONDS, RANK_9),
        cid(SUIT_DIAMONDS, RANK_8), cid(SUIT_CLUBS, RANK_9),
        cid(SUIT_CLUBS, RANK_8), cid(SUIT_CLUBS, RANK_7),
        cid(SUIT_CLUBS, RANK_6),
    ]
    # P3: нет пик, но есть козыри 9♥ и 7♥ -> обязан бить 9♥ => нет пик И нет 10♥..A♥
    p3_cards = [
        cid(SUIT_HEARTS, RANK_9), cid(SUIT_HEARTS, RANK_7), cid(SUIT_HEARTS, RANK_6),
        cid(SUIT_DIAMONDS, RANK_7), cid(SUIT_DIAMONDS, RANK_6),
        cid(SUIT_HEARTS, RANK_8), cid(SUIT_SPADES, RANK_6),  # заменим 6♠ на 10♥? Нет, 6♠ нельзя P3!
        JK2,
    ]
    # Сделаем строго без пик у P3 и с upcard = 10♥
    p3_cards = [
        cid(SUIT_HEARTS, RANK_9), cid(SUIT_HEARTS, RANK_8), cid(SUIT_HEARTS, RANK_7), cid(SUIT_HEARTS, RANK_6),
        cid(SUIT_DIAMONDS, RANK_7), cid(SUIT_DIAMONDS, RANK_6),
        cid(SUIT_HEARTS, RANK_10),  # заменим на JK2, а 10♥ отдадим в upcard!
        JK2,
    ]
    # Точный бездубликатный расклад (по 9 карт на 4 игроков + 1 upcard = 37 карт из 38):
    p0_cards = [
        cid(SUIT_SPADES, RANK_A), cid(SUIT_SPADES, RANK_K),
        cid(SUIT_HEARTS, RANK_A), cid(SUIT_HEARTS, RANK_K),
        cid(SUIT_DIAMONDS, RANK_A), cid(SUIT_DIAMONDS, RANK_K),
        cid(SUIT_CLUBS, RANK_A), cid(SUIT_CLUBS, RANK_K),
        cid(SUIT_CLUBS, RANK_Q),
    ]
    p1_cards = [
        JK1,
        cid(SUIT_SPADES, RANK_Q), cid(SUIT_SPADES, RANK_J),
        cid(SUIT_HEARTS, RANK_Q), cid(SUIT_HEARTS, RANK_J),
        cid(SUIT_DIAMONDS, RANK_Q), cid(SUIT_DIAMONDS, RANK_J),
        cid(SUIT_CLUBS, RANK_J), cid(SUIT_CLUBS, RANK_10),
    ]
    p2_cards = [
        cid(SUIT_SPADES, RANK_10), cid(SUIT_SPADES, RANK_9), cid(SUIT_SPADES, RANK_8),
        cid(SUIT_SPADES, RANK_7), cid(SUIT_SPADES, RANK_6),
        cid(SUIT_DIAMONDS, RANK_10), cid(SUIT_DIAMONDS, RANK_9),
        cid(SUIT_CLUBS, RANK_9), cid(SUIT_CLUBS, RANK_8),
    ]
    p3_cards = [
        cid(SUIT_HEARTS, RANK_9), cid(SUIT_HEARTS, RANK_8), cid(SUIT_HEARTS, RANK_7), cid(SUIT_HEARTS, RANK_6),
        cid(SUIT_DIAMONDS, RANK_8), cid(SUIT_DIAMONDS, RANK_7), cid(SUIT_DIAMONDS, RANK_6),
        cid(SUIT_CLUBS, RANK_7), cid(SUIT_CLUBS, RANK_6),
    ]
    upcard = cid(SUIT_HEARTS, RANK_10)  # козырь Червы (37-я карта; JK2 в остатке 38-й картой)

    deck = make_custom_deck(p0_cards + p1_cards + p2_cards + p3_cards + [upcard], deck_size=38)
    lib.joker_force_deck_order(env, deck)
    lib.joker_reset(env, RULES_CLASSIC, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, 7)

    # Торговля (P1, P2, P3, P0)
    for b in (2, 2, 2, 2):
        assert lib.joker_step(env, b) == 1

    # Взятка 1: P1 ходит JK1 LeadHighSuit(Spades)
    assert lib.joker_step(env, ACT_JOKER_LEAD_HIGH_BASE + SUIT_SPADES) == 1
    # P2 кладёт старшую пику 10♠
    assert lib.joker_step(env, cid(SUIT_SPADES, RANK_10)) == 1
    # P3 (нет пик) кладёт старший козырь 9♥
    assert lib.joker_step(env, cid(SUIT_HEARTS, RANK_9)) == 1
    # P0 кладёт старшую пику A♠
    assert lib.joker_step(env, cid(SUIT_SPADES, RANK_A)) == 1

    forbidden_p2 = mask_of(
        cid(SUIT_SPADES, RANK_J), cid(SUIT_SPADES, RANK_Q),
        cid(SUIT_SPADES, RANK_K), cid(SUIT_SPADES, RANK_A),
    )
    forbidden_p3 = (
            mask_of(*[cid(SUIT_SPADES, r) for r in range(9)])
            | mask_of(
        cid(SUIT_HEARTS, RANK_10), cid(SUIT_HEARTS, RANK_J),
        cid(SUIT_HEARTS, RANK_Q), cid(SUIT_HEARTS, RANK_K), cid(SUIT_HEARTS, RANK_A),
    )
    )

    # Проверяем 100 детерминизаций от лица P0 (P0 не видит карты P1/P2/P3)
    for s in range(1, 101):
        det = lib.joker_clone_determinized(env, 0, s)
        try:
            ds = get_snap(det)
            assert (ds.hands[2] & forbidden_p2) == 0, f"P2 получил невозможную старшую пику (seed={s})"
            assert (ds.hands[3] & forbidden_p3) == 0, f"P3 получил пику или старший козырь (seed={s})"
            for p in range(4):
                assert popcount(ds.hands[p]) == 8
                assert (ds.hands[p] & ds.played_cards) == 0
                assert (ds.hands[p] & (1 << upcard)) == 0
        finally:
            lib.joker_destroy(det)


def test_observed_mode_nines_errors_undo_history_and_fork(make_env, make_vec_env):
    """Полный сценарий живого стола: 3->козырь->6, JokerStepError, Undo, history_tokens и Fork."""
    env = make_env(rules=RULES_POPULAR, mode=MODE_SHORT, team=TEAM_SOLO, pulka=PULKA_STRICT, seed=1)
    lib.joker_begin_observed_match(env, RULES_POPULAR, MODE_SHORT, TEAM_SOLO, PULKA_STRICT, 0, 1)

    # 1. До ввода карт шаг запрещён
    assert lib.joker_get_legal_actions_mask(env) == 0
    assert lib.joker_validate_action(env, SUIT_HEARTS) == JOKER_ERR_WAITING_FOR_DEAL_INPUT

    # 2. Ввод неверного числа карт отклонён
    first_three = mask_of(cid(SUIT_HEARTS, RANK_A), cid(SUIT_HEARTS, RANK_K), cid(SUIT_SPADES, RANK_A))
    assert lib.joker_set_observed_deal(env, mask_of(cid(SUIT_HEARTS, RANK_A)), INVALID_CARD, SUIT_NONE) == 0
    assert lib.joker_set_observed_deal(env, first_three, INVALID_CARD, SUIT_NONE) == 1
    assert lib.joker_get_phase(env) == PHASE_TRUMP
    assert lib.joker_get_current_player(env) == 1

    # 3. Выбор козыря -> ожидание дораздачи 6 карт
    assert lib.joker_step_checked(env, SUIT_HEARTS) == JOKER_ERR_OK
    assert lib.joker_validate_action(env, 0) == JOKER_ERR_WAITING_FOR_DEAL_INPUT

    # 4. Проверка подмножества 3 карт при дораздаче 6
    bad_nine = mask_of(
        cid(SUIT_HEARTS, RANK_Q), cid(SUIT_HEARTS, RANK_J), cid(SUIT_HEARTS, RANK_10),
        cid(SUIT_DIAMONDS, RANK_A), cid(SUIT_DIAMONDS, RANK_K), cid(SUIT_DIAMONDS, RANK_Q),
        cid(SUIT_CLUBS, RANK_A), cid(SUIT_CLUBS, RANK_K), cid(SUIT_CLUBS, RANK_Q),
    )
    assert lib.joker_set_observed_remaining_six(env, bad_nine) == 0

    full_nine = first_three | mask_of(
        cid(SUIT_HEARTS, RANK_Q), cid(SUIT_DIAMONDS, RANK_A), cid(SUIT_DIAMONDS, RANK_K),
        cid(SUIT_CLUBS, RANK_A), cid(SUIT_CLUBS, RANK_K), cid(SUIT_SPADES, RANK_K),
    )
    assert lib.joker_set_observed_remaining_six(env, full_nine) == 1
    assert lib.joker_get_phase(env) == PHASE_BIDDING

    # 5. Торговля и проверка JOKER_ERR_FORBIDDEN_DEALER_BID
    assert lib.joker_step_checked(env, 3) == JOKER_ERR_OK  # P1
    assert lib.joker_step_checked(env, 3) == JOKER_ERR_OK  # P2
    assert lib.joker_step_checked(env, 3) == JOKER_ERR_OK  # P3 (сумма = 9 => дилеру P0 запрещён 0)
    assert lib.joker_step_checked(env, 0) == JOKER_ERR_FORBIDDEN_DEALER_BID
    assert lib.joker_step_checked(env, 1) == JOKER_ERR_OK  # P0

    assert lib.joker_get_phase(env) == PHASE_PLAY
    assert lib.joker_get_current_player(env) == 1

    # 6. Розыгрыш: герой ходит A♠, затем тестируем ошибки соперника P2
    assert lib.joker_step_checked(env, cid(SUIT_SPADES, RANK_A)) == JOKER_ERR_OK
    assert lib.joker_get_current_player(env) == 2

    # Карта уже вышла
    assert lib.joker_step_checked(env, cid(SUIT_SPADES, RANK_A)) == JOKER_ERR_CARD_ALREADY_PLAYED
    # Карта в известной руке героя
    assert lib.joker_step_checked(env, cid(SUIT_HEARTS, RANK_A)) == JOKER_ERR_CARD_IN_OTHER_KNOWN_HAND

    # P2 делает ренонс (кладет 7♦ при ходе в пику -> нет ни пик, ни козыря ♥)
    assert lib.joker_step_checked(env, cid(SUIT_DIAMONDS, RANK_7)) == JOKER_ERR_OK
    assert lib.joker_step_checked(env, cid(SUIT_SPADES, RANK_7)) == JOKER_ERR_OK  # P3
    assert lib.joker_step_checked(env, cid(SUIT_SPADES, RANK_8)) == JOKER_ERR_OK  # P0 -> взятку взял P1 (A♠)

    # Взятка 2: P1 ходит K♠, P2 пытается сыграть 9♠ — но у него уже зафиксирован ренонс по пикам!
    assert lib.joker_step_checked(env, cid(SUIT_SPADES, RANK_K)) == JOKER_ERR_OK
    assert lib.joker_step_checked(env, cid(SUIT_SPADES, RANK_9)) == JOKER_ERR_VIOLATES_KNOWN_VOID_OR_BOUND

    # 7. История сдачи в Single-Env
    tokens, length = get_history_tokens(env, 1)
    assert length == 1 + 4 + 4 + 1  # 1 trump + 4 bids + 4 plays + 1 play
    assert tokens[0, 0] == 1  # ChoosingTrump

    # 8. Откат хода (Undo)
    undo_before = lib.joker_can_undo_steps(env)
    assert undo_before >= 10
    assert lib.joker_undo(env) == 1
    assert lib.joker_get_current_player(env) == 1
    _, length_after_undo = get_history_tokens(env, 1)
    assert length_after_undo == length - 1

    # 9. Fork в JokerVectorEnv прямо из Observed Mode (для подсказки бота)
    vec = make_vec_env(num_envs=16, rules=RULES_POPULAR, mode=MODE_SHORT, threads=2)
    obs = np.zeros((16, lib.joker_tensor_size()), dtype=np.float32)
    masks = np.zeros(16, dtype=np.uint64)
    oracle = np.zeros((16, lib.joker_oracle_tensor_size()), dtype=np.float32)
    lib.joker_vec_fork_from_env(
        vec, env, 1, 777,
        obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        oracle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    assert np.all(masks == lib.joker_get_legal_actions_mask(env))
    # У каждого из 4 игроков в детерминизированных мирах по 8 карт
    for i in range(16):
        o = oracle[i]
        for rel in range(4):
            assert int(np.sum(o[rel * 38 : (rel + 1) * 38])) == 8