# Joker Core Engine (`C++23`)

Высокопроизводительное, тривиально копируемое и кроссплатформенное ядро карточной игры **«Джокер»** на стандарте **C++23** с нулевыми аллокациями в горячем цикле, многопоточной векторизацией (**OpenMP**) и чистым **C ABI** (`Zero-Copy`).

Подходит для **живого стола (Live Assistant)**, **массового RL-обучения (PPO / IMPALA / MAPPO / AlphaZero)**, **гибридного поиска (PIMC / IS-MCTS)** и **турнирной арены ботов**.

---

## Ключевые особенности

* **Zero-Allocation & Trivially Copyable Core (`std::is_trivially_copyable_v<JokerEngine> == true`):**
  Весь конечный автомат партии (`GameState` 192 байта + `Deck` с `xoshiro256**` и Lemire unbiased shuffle) занимает **320 байт** без динамической памяти. `EnvSlot` и `SlotStats` выровнены по 64 байтам (`alignas(64)`).
* **24 ортогональные конфигурации правил (`3 × 2 × 2 × 2`):**
  * `RulePreset`: `Popular`, `Minus`, `Classic` (36 или 38 карт);
  * `MatchMode`: `Long` (24 сдачи) и `Short` (8 сдач по 9 карт);
  * `TeamMode`: `Solo` и `Pairs` (`P0+P2` vs `P1+P3`);
  * `PulkaMode`: `Strict` и `Soft` (сосед «над вами» `over = (P + 3) % 4`).
* **Режим живого стола (Observed Mode) + вход в середину партии + `Undo`:**
  Поддержка частичной наблюдаемости (неизвестные руки соперников), старт с произвольной сдачи (`joker_set_observed_match_state`), флаги ожидания ввода в `JokerSnapshot` (`observed_flags`, `hero_seat`), валидация правил с 15 кодами `JokerStepError`, откат на 64 шага (`joker_undo`) и `DealHistoryBuffer` прямо в одиночной среде.
* **Жёсткая логическая память (`impossibleCards`) и точная детерминизация:**
  Отслеживание ренонсов и потолков рангов после хода Джокером `LeadHighSuit`; 3-проходная безаллокационная детерминизация (включая BFS по дополняющим циклам) в модуле `Determinizer`.
* **Входной тензор v2 (`320 float`) + Oracle (`160 float`) + History (`[48, 8] int16`):**
  Компактное SIMD-выровненное представление без абсолютных счетов — марковский горизонт будущего, контрфактические спреды пульки, потолки рангов и счётчик `master cards`.
* **Встроенная C++ эвристика (`HeuristicBot`) и Арена (`JokerPlayerStats`):**
  Усиленный 3-фазный бот с учётом `impossibleCards`, экономики пульки, парной координации и тактики подсада (винрейт > 90% против случайного игрока во всех 3 пресетах, производительность > 80 000 партий/сек в OpenMP), сбор полной статистики без блокировок и зеркальные турниры (`joker_vec_reset_with_seeds`, `joker_vec_set_auto_reset`).

---

## Сборка (Desktop / Android / iOS / WebAssembly)

Требования: компилятор с **C++23** (`GCC 13+`, `Clang 16+`, `MSVC 2022+`) и **CMake 3.20+**.

### Desktop (Linux / macOS / Windows)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/joker_cli
```

### Кроссплатформенные опции CMake
| Опция | По умолчанию | Назначение |
| :--- | :---: | :--- |
| `JOKER_ENABLE_OPENMP` | `ON` (`OFF` на iOS/WASM) | Многопоточность OpenMP в `JokerVectorEnv` |
| `JOKER_NATIVE_ARCH` | `ON` (`OFF` при кросс-сборке) | Флаги `-march=native` / `/arch:AVX2` |
| `JOKER_BUILD_SHARED` | `ON` (`OFF` на iOS/WASM) | Разделяемая (`.so`/`.dll`/`.dylib`) или статическая библиотека |
| `JOKER_BUILD_CLI` | `ON` (`OFF` на мобильных/WASM) | Сборка утилиты `joker_cli` |

Примеры кросс-сборки:
```bash
# Портативная сборка без -march=native
cmake -B build-portable -DCMAKE_BUILD_TYPE=Release -DJOKER_NATIVE_ARCH=OFF
cmake --build build-portable -j

# WebAssembly (Emscripten)
emcmake cmake -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm -j
```

---

## Быстрый старт (Python + NumPy)

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("./build/libjoker_core.so")
lib.joker_tensor_size.restype = ctypes.c_size_t
TENSOR_DIM = lib.joker_tensor_size()  # 320

NUM_ENVS = 1024
vec = lib.joker_vec_create(NUM_ENVS, 0, 0, 0, 0, 42, 0)  # Popular, Long, Solo, Strict

obs = np.zeros((NUM_ENVS, TENSOR_DIM), dtype=np.float32)
masks = np.zeros(NUM_ENVS, dtype=np.uint64)
players = np.zeros(NUM_ENVS, dtype=np.uint8)
rewards = np.zeros((NUM_ENVS, 4), dtype=np.float32)
deal_dones = np.zeros(NUM_ENVS, dtype=np.uint8)
match_dones = np.zeros(NUM_ENVS, dtype=np.uint8)
scores = np.zeros((NUM_ENVS, 4), dtype=np.int32)

lib.joker_vec_get_observations(
    vec,
    obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    players.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    None, None, None,
)

actions = np.zeros(NUM_ENVS, dtype=np.uint8)  # выберите легальные биты из masks
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
    None, None, None,
)

lib.joker_vec_destroy(vec)
```

---

## Тестирование и бенчмарки

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest numpy

# Юнит-тесты без бенчмарков
pytest -v -m "not perf"

# Полный прогон вместе с бенчмарком производительности
pytest -v -s
```

---

## Документация

* **[ENGINE.md](ENGINE.md)** — спецификация модулей, канонических `ActionId` (`0..45`), раскладки тензора v2 (`320 float`), Oracle (`160 float`), истории (`[48, 8] int16`), Observed Mode и всего `joker_c_api.h`.
* **[RULES.md](RULES.md)** — исчерпывающее описание правил игры «Джокер» для всех режимов.