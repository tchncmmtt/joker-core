# Joker Core Engine (`C++23`) — Архитектура и документация API

Высокопроизводительное ядро карточной игры «Джокер» на стандарте **C++23**, спроектированное для:
* **живого стола (Observed / Live Table Mode):** ввод реальных раздач, строгая валидация правил с кодами ошибок (`JokerStepError`), откат (`Undo`) и подсказка хода от любого бота;
* **массового обучения с подкреплением** (PPO, IMPALA, MAPPO, AlphaZero) через многопоточный батчинг (**OpenMP**) с производительностью ≥ 10 000 партий/сек;
* **нейросетевых архитектур** (**MLP, ResNet, LSTM, Transformers**) и **Asymmetric Actor-Critic**;
* **арены и бенчмаркинга ботов:** встроенная C++ эвристика (`HeuristicBot`), сбор статистики (`JokerPlayerStats`) без блокировок и зеркальные турниры (Duplicate Matches);
* **гибридного поиска** (**PIMC / IS-MCTS + Batch NN**) с точной детерминизацией по матрице `impossibleCards`.

---

## 1. Архитектура и структура проекта

```text
joker-core/
├── include/
│   ├── core/
│   │   ├── Types.hpp          # Константы, кросс-макросы, режимы игры и StepError
│   │   ├── Card.hpp           # Таблица карт (0..37), битовые маски мастей и старших карт
│   │   ├── Action.hpp         # Каноническое пространство действий (0..45)
│   │   ├── Deck.hpp           # xoshiro256** + тасование Лемира без modulo bias
│   │   ├── Rules.hpp          # Легальные ходы, валидация, взятки и подсчёт очков
│   │   └── GameState.hpp      # 192-байтное состояние партии и тензор v2 (320 float)
│   ├── engine/
│   │   ├── JokerEngine.hpp    # Тривиально копируемый конечный автомат партии
│   │   ├── HeuristicBot.hpp   # Быстрая 3-фазная безаллокационная C++ эвристика
│   │   └── JokerVectorEnv.hpp # OpenMP-пул сред, DealHistoryBuffer, Oracle, Fork, Stats
│   └── joker_c_api.h          # Чистый C ABI (Zero-Copy) для Python / Rust / C# / WASM
├── src/
│   ├── core/
│   │   ├── Card.cpp
│   │   ├── Deck.cpp
│   │   └── GameState.cpp
│   ├── engine/
│   │   ├── HeuristicBot.cpp
│   │   ├── JokerEngine.cpp
│   │   └── JokerVectorEnv.cpp
│   ├── joker_c_api.cpp
│   └── main_cli.cpp
└── CMakeLists.txt
```

### Ключевые инженерные решения
1. **Тривиально копируемое ядро (`std::is_trivially_copyable_v<JokerEngine> == true`):**
   Весь автомат партии (`GameState` + `Deck` с `xoshiro256**`) занимает **320 байт** без динамических аллокаций. Клонирование узла в MCTS выполняется за один `memcpy`. Буферы `Undo` (64 снимка) и `DealHistoryBuffer` живут в обёртке `JokerEnvHandle`, не раздувая `JokerEngine`.
2. **Выравнивание под кэш-линии процессора:**
   * `GameState` занимает ровно **192 байта** (`alignas(64)`, 3 кэш-линии L1).
   * `EnvSlot` и `SlotStats` в `JokerVectorEnv` выровнены по границе **64 байт** (`alignas(64)`), что исключает *False Sharing* в OpenMP.
3. **Zero-Copy C ABI:**
   Все функции C API пишут тензоры, маски, токены и статистику напрямую в буферы вызывающей стороны без аллокаций в горячем цикле.

---

## 2. Ортогональные режимы конфигурации

Движок поддерживает **24 независимые комбинации** (`3 × 2 × 2 × 2`):

| Параметр | Перечисление C++ | Значения в C API | Описание |
| :--- | :--- | :--- | :--- |
| **Пресет правил** | `RulePreset` | `0 = Popular`<br>`1 = Minus`<br>`2 = Classic` | Колода 36 или 38 карт, козырь на девятках (гадание / вскрытие), формулы недобора и штанги |
| **Длина партии** | `MatchMode` | `0 = Long`<br>`1 = Short` | `Long`: 24 сдачи (4 пульки `8+4+8+4`)<br>`Short`: 8 сдач (2 пульки `4+4` строго по 9 карт) |
| **Командный режим** | `TeamMode` | `0 = Solo`<br>`1 = Pairs` | `Solo`: каждый за себя<br>`Pairs`: пара `P0 + P2` против `P1 + P3` |
| **Расчёт пульки** | `PulkaMode` | `0 = Strict`<br>`1 = Soft` | `Strict`: классическое списание и отмена премии<br>`Soft`: соседский расчёт по игроку «над вами» `over = (P + 3) % 4` (сосед справа, ходит перед вами) |

---

## 3. Кодирование карт и каноническое пространство действий (`ActionId`)

### 3.1. Индексы карт (`CardId: 0..37`)
* `0 .. 8` — Пики (`♠`): `6♠ (0), 7♠ (1), ..., A♠ (8)`. В `Popular` и `Minus` бит `0` (`6♠`) не используется.
* `9 .. 17` — Трефы (`♣`): `6♣ (9), 7♣ (10), ..., A♣ (17)`. В `Popular` и `Minus` бит `9` (`6♣`) не используется.
* `18 .. 26` — Бубны (`♦`): `6♦ (18), ..., A♦ (26)`.
* `27 .. 35` — Червы (`♥`): `6♥ (27), ..., A♥ (35)`.
* `36` — Джокер 1 (`JK1`), `37` — Джокер 2 (`JK2`).

### 3.2. Унифицированное пространство действий (`0..45`, всего 46 действий)
Каноническая нумерация задаётся в `include/core/Action.hpp`:

| Фаза (`Phase`) | Диапазон `ActionId` | Семантика действия |
| :--- | :---: | :--- |
| **`ChoosingTrump` (`0`)** | `0 .. 4` | Выбор козыря: `0 = ♠`, `1 = ♣`, `2 = ♦`, `3 = ♥`, `4 = Бескозырка (None)` |
| **`Bidding` (`1`)** | `0 .. 9` | Заказ взяток (`0` = пас, `1..9` = количество взяток; для дилера запрещённый «висяк» обнулён) |
| **`TrickPlay` (`2`) — Обычная карта** | `0 .. 35` | Ход обычной картой с соответствующим `CardId` |
| **`TrickPlay` (`2`) — Ответ Джокером** | `36`<br>`37` | `36`: Джокер «побить / забрать» (`PlayWin`)<br>`37`: Джокер «сбросить по дну» (`PlayLow`) |
| **`TrickPlay` (`2`) — Первый ход Джокером** | `38 .. 41`<br>`42 .. 45` | `38..41`: Джокер «по высшей» (`LeadHighSuit`) на масть `0..3` (`♠, ♣, ♦, ♥`)<br>`42..45`: Джокер «по меньшей» (`LeadLow`) на масть `0..3` (`♠, ♣, ♦, ♥`) |

---

## 4. Форматы входных тензоров и буферов

### 4.1. Входной тензор v2 (`320 float`, `10 × 32 float = 1280` байт)
Тензор строится **эгоцентрично** относительно `observer` (`0` = Я, `1` = Слева / враг под нами, `2` = Напротив / партнёр, `3` = Справа / враг над нами). **Абсолютные счета полностью исключены** — используются только относительные отрывы и марковский горизонт будущего.

| Блок | Индексы | Размер | Содержание |
| :--- | :---: | :---: | :--- |
| **A. Фаза и горизонт будущего** | `0 .. 17` | 18 | Единый для `Long` и `Short` |
| **B. Относительный счёт и пулька** | `18 .. 41` | 24 | Раздельный для `Solo` и `Pairs` |
| **C. Контракты и текущая взятка** | `42 .. 119` | 78 | Контракты, баланс стола, сила взятки |
| **D. Карты, память и Master Cards** | `120 .. 319` | 200 | 4 маски по 38 карт, `impossibleCards`, счёт карт |

#### Блок A (`0 .. 17`) — Фаза и горизонт будущего
* `0..2`: `Phase` one-hot (`ChoosingTrump`, `Bidding`, `TrickPlay`).
* `3`: `cardsInDeal / 9.0`; `4`: `remainingTricksInDeal / 9.0`.
* `5..8`: `relDealer` one-hot (`0..3`); `9`: `relBiddingOrder / 3.0` (`0.0` = первое слово, `1.0` = дилер).
* `10`: `remainingDealsInPulka / 8.0`; `11`: `remainingCardsInPulka / 36.0`.
* `12`: `isPulkaEnd`; `13`: `isNinesPulka`.
* `14`: `nextDealCards / 9.0` (`0.0` при `isPulkaEnd`); `15`: `maxFutureNonLastDealCards / 9.0`.
* `16`: `remainingPulkasAfterCurrent / 3.0`; `17`: `remainingMatchCards / 144.0` (`1.0→0.0` в `Long`, `0.5→0.0` в `Short`).

#### Блок B (`18 .. 41`) — Относительный счёт и контрфактическая экономика пульки
* **Для `Solo` (`18..25`):**
  * `18..20`: `(score[opp_1..3] - score[me]) / 500.0`.
  * `21..23`: `(score[opp_1..3] - score[me]) / max(100.0, remainingMatchCards * 25.0)`.
  * `24`: `myRankNorm` (`1.0` = 1-е место, `0.0` = 4-е); `25`: резерв (`0.0`).
* **Для `Pairs` (`18..25`):**
  * `18`: `(enemyTeamScore - myTeamScore) / 500.0`.
  * `19`: `(enemyTeamScore - myTeamScore) / max(100.0, remainingMatchCards * 50.0)`.
  * `20`: `isMyTeamLeading`; `21..22`: `myTeamClean / 2.0`, `enemyTeamClean / 2.0`.
  * `23..24`: `projectedMyTeamPulkaDelta / 500.0`, `projectedEnemyTeamPulkaDelta / 500.0`; `25`: резерв (`0.0`).
* **Общие слоты пульки (`26..41`, учитывают `Strict`/`Soft` и `Solo`/`Pairs`):**
  * `26..29`: `pulkaClean[0..3]`; `30..33`: `pulkaAllPasses[0..3]`.
  * `34..37`: `projectedPulkaDelta[0..3] / 500.0` — точная премия/штраф при сохранении текущих статусов.
  * `38..41`: **контрфактические спреды** `failureNetSwing[0..3] / 500.0` — изменение относительного баланса `observer` (против среднего соперника в `Solo` или против чужой пары в `Pairs`), если конкретный игрок `rel` сорвёт текущую сдачу.

#### Блок C (`42 .. 119`) — Контракты и текущая взятка
* `42..65` (`6 × 4` по `rel = 0..3`): `bid / 9.0` (`42..45`), `hasBid` (`46..49`), `tricksTaken / 9.0` (`50..53`), `deficitRatio = clamp(need / max(1, remTricks), 0, 1.5)` (`54..57`), `isContractBroken` (`58..61`), `isShangaRisk` (`62..65`).
* `66..69`: баланс стола `(cardsInDeal - sumBids) / 9.0`, `sumBids / cardsInDeal`; в `Pairs` дополнительно `myTeamBid / 18.0`, `myTeamNeed / 18.0`.
* `70..79`: запрещённый «висяк» дилера one-hot (`0..9`).
* `80..84`: `trumpSuit` one-hot (`5`); `85..89`: `requiredSuit` one-hot (`5`).
* `90`: `cardsInCurrentTrick / 3.0`; `91`: `forceHigh`; `92`: `isLead`.
* `93..96`: `currentWinner` one-hot (`4`); `97`: `partnerWinning` (в `Pairs`).
* `98`: `trickHeldByJokerWin`; `99`: `trickHeldByTrump`; `100`: `currentWinningRank / 8.0`.
* `101..104`: `relTrickLeader` one-hot (`4`).
* `105..116`: действия Джокера в 3 слотах взятки (`3 × 4`).
* `117..119`: резерв (`0.0`).

#### Блок D (`120 .. 319`) — Карты, логическая память и Master Cards
* `120..157`: `myHand` (38); `158..195`: `playedCardsHistory` (38); `196..233`: `upcardMask` (38); `234..271`: объединённая `currentTrickMask` (38).
* `272..283`: явные ренонсы оппонентов (`3 × 4`).
* `284..295`: **потолок максимального возможного ранга** оппонентов (`3 × 4`, `(maxPossibleRank + 1) / 9.0` с учётом `impossibleCards` после `LeadHighSuit`, `0.0` при ренонсе).
* `296..299`: `jokersPlayedBy[0..3] / 2.0`; `300`: `unseenJokers / 2.0`; `301`: `myJokers / 2.0`.
* `302..305`: `unseenInSuit / maxSuit`; `306..309`: `myInSuit / maxSuit`.
* `310..313`: **`myMasterCardsInSuit / 4.0`** (число старших небитых карт на руке в масти).
* `314..317`: **`highestUnseenOppRankInSuit / 9.0`** (ранг старшей невышедшей чужой карты).
* `318`: `winningCardsInMyHand / max(1, handSize)`; `319`: `losingCardsInMyHand / max(1, handSize)`.

### 4.2. Oracle-тензор (`160 float`) и история сдачи (`[48, 8] int16`)
* **Oracle (`160 float`):** `0..151` (`4 × 38`) — истинные карты всех 4 игроков в эгоцентричном порядке; `152..159` — выравнивание.
* **DealHistoryBuffer (`[48, 8] int16`):** хронологическая последовательность событий сдачи (доступна и в `JokerVectorEnv`, и в Single-Env через `joker_write_history_tokens`). Значение `0` во всех 8 колонках строго зарезервировано под `PAD`:
  `[eventType(1..3), relSeat(1..4), actionId+1(1..46), cardId+1(0..38), suit+1(0..5), jokerAction+1(0..5), trickIdx+1(0..9), posInTrick+1(0..4)]`.

---

## 5. Логическая память (`impossibleCards`) и детерминизация

В `GameState` хранится битовая матрица `impossibleCards[4]` (38 бит на игрока):
* при ренонсе по требуемой масти исключается вся масть (и вся козырная масть, если игрок не бил козырем);
* при ходе Джокером `LeadHighSuit` все карты требуемой масти (или козыря, если игрок бил козырем при ренонсе) **строго старше** сыгранной карты заносятся в `impossibleCards[p]`.

`JokerEngine::Determinize(observer, seed)` использует `impossibleCards` и счётчики `handCounts[4]` (включая Observed Mode) через 3-проходный безаллокационный солвер: жадный обмен → парный обмен → точный **BFS по дополняющим циклам** среди 4 корзин на стеке.

---

## 6. Режим живого стола (Observed Mode), валидация и `Undo`

Протокол ведения реального стола через `JokerEnvHandle`:
1. `joker_begin_observed_match(env, rules, mode, team, pulka, dealer, hero_seat)` — переводит движок в частичную наблюдаемость (`knownHandsMask = 1 << hero_seat`).
2. `joker_set_observed_deal(env, hero_hand_mask, upcard_or_invalid, trump_suit_or_none)` — ввод карт героя (3 карты на девятках с гаданием или полную руку) и вскрытого козыря.
3. На девятках после `ChoosingTrump` вызывается `joker_set_observed_remaining_six(env, full_nine_hand_mask)`.
4. Ходы проверяются через `joker_validate_action` / `joker_step_checked`, возвращающие `JokerStepError` (`0 = JOKER_ERR_OK` .. `14 = JOKER_ERR_NO_JOKER_AVAILABLE`).
5. Любой ошибочный ввод откатывается через `joker_undo(env)` (кольцевой буфер на 64 снимка, глубина — `joker_can_undo_steps`).
6. В любой момент `joker_vec_fork_from_env` клонирует живой стол в пул `JokerVectorEnv` с детерминизацией рук соперников и копированием `DealHistoryBuffer` для подсказки нейросети/MCTS.

---

## 7. Справочник C API (`joker_c_api.h`)

### Одиночная среда (`JokerEnvHandle`)
* **Жизненный цикл:** `joker_create`, `joker_destroy`, `joker_clone`, `joker_clone_determinized`, `joker_determinize_inplace`, `joker_reset`, `joker_reset256`, `joker_force_deck_order`.
* **Observed Mode:** `joker_begin_observed_match`, `joker_set_observed_deal`, `joker_set_observed_remaining_six`.
* **Шаг, валидация, Undo:** `joker_step`, `joker_validate_action`, `joker_step_checked`, `joker_undo`, `joker_can_undo_steps`.
* **Наблюдения и бот:** `joker_get_legal_actions_mask`, `joker_write_observation_tensor`, `joker_write_history_tokens`, `joker_select_heuristic_action`, `joker_get_snapshot`, `joker_get_total_scores`, `joker_get_team_scores`.
* **Размерности:** `joker_tensor_size() == 320`, `joker_oracle_tensor_size() == 160`, `joker_history_seq_len() == 48`, `joker_history_token_dim() == 8`.

### Векторизованная среда и Арена (`JokerVecEnvHandle`)
* **Управление пулом:** `joker_vec_create`, `joker_vec_destroy`, `joker_vec_size`, `joker_vec_reset`, `joker_vec_reset_with_seeds` (Duplicate Matches), `joker_vec_set_auto_reset`, `joker_vec_set_reward_config`.
* **Статистика Арены:** `joker_vec_set_stats_enabled` (по умолчанию выключена для RL), `joker_vec_reset_stats`, `joker_vec_get_stats(vec, out_stats4)`.
* **Симуляция и Fork:** `joker_vec_fork_from_env`, `joker_vec_get_observations`, `joker_vec_step`, `joker_vec_advance_to_hero`, `joker_vec_step_arena`.