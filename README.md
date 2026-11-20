# Joker Core Engine (`C++23`)

Высокопроизводительное, тривиально копируемое ядро карточной игры **«Джокер»** на стандарте **C++23** с нулевыми аллокациями в горячем цикле, многопоточной векторизацией (**OpenMP**) и чистым **C ABI** (`Zero-Copy`) для обучения с подкреплением и гибридного поиска (**PPO / IMPALA / MAPPO / AlphaZero / PIMC / IS-MCTS**).

---

## Ключевые особенности

* **Zero-Allocation & Trivially Copyable Core (`std::is_trivially_copyable_v<JokerEngine> == true`):**
  Весь конечный автомат партии, состояние ГПСЧ `xoshiro256**` и колода занимают ~180 байт без динамической памяти (`new` / `std::vector`). Клонирование состояния для MCTS-симуляций сводится к одному `memcpy`.
* **Cache-Line Aligned:**
  * `GameState` выровнен по границе **128 байт** (`alignas(64)`, две кэш-линии L1).
  * Слоты в `JokerVectorEnv` выровнены по **64 байта** (`alignas(64)`), что полностью исключает *False Sharing* при параллельном исполнении в потоках OpenMP.
* **Zero-Copy C ABI:**
  Тензоры наблюдений, битовые маски легальных ходов, Oracle-данные и токены истории пишутся напрямую в предварительно выделенные буферы NumPy / `torch.Tensor`.
* **24 ортогональные конфигурации правил (`3 × 2 × 2 × 2`):**
  * **Пресеты правил (`RulePreset`):** `Popular` (36 карт, гадание на девятках, «все свои» 100×, штанга -200), `Minus` (36 карт, строгий штраф за недобор -100×), `Classic` (38 карт со всеми шестёрками, вскрытие козыря на девятках).
  * **Длина партии (`MatchMode`):** `Long` (24 сдачи, 4 пульки `8+4+8+4`) и `Short` (8 сдач, 2 пульки `4+4` по 9 карт).
  * **Командный режим (`TeamMode`):** `Solo` (каждый за себя) и `Pairs` (`P0 + P2` против `P1 + P3`).
  * **Расчёт пульки (`PulkaMode`):** `Strict` (классическое списание и отмена премии) и `Soft` (соседский расчёт по игрокам «над» и «под» вами).
* **Поддержка любых нейросетевых архитектур:**
  * **MLP / ResNet:** эгоцентричный вектор наблюдения `384 float` (с раздельной экономикой для `Solo` и `Pairs`).
  * **LSTM / Transformers:** хронологический буфер событий сдачи `[48, 8] int16` с зарезервированным `PAD = 0`.
  * **Asymmetric Actor-Critic:** скрытый Oracle-тензор `160 float` с истинными картами всех 4 игроков.
* **Гибридный поиск (PIMC / IS-MCTS + Batch NN):**
  Встроенная детерминизация скрытых карт (`joker_clone_determinized`, `joker_vec_fork_from_env`) с гарантированным соблюдением вышедших карт, прикупа и известных ренонсов (`voidSuits`).

---

## Структура репозитория

```text
joker-core/
├── include/
│   ├── core/
│   │   ├── Types.hpp          # Константы и 4 ортогональных режима игры
│   │   ├── Card.hpp           # Таблица карт (0..37) и битовые маски мастей
│   │   ├── Action.hpp         # Пространство действий (0..45) и кодирование ходов
│   │   ├── Deck.hpp           # xoshiro256** + тасование Лемира без modulo bias
│   │   ├── Rules.hpp          # Легальные ходы, старшинство взяток (с Джокерами) и очки
│   │   └── GameState.hpp      # 128-байтное состояние партии и проекция наблюдения (384 float)
│   ├── engine/
│   │   ├── JokerEngine.hpp    # Тривиально копируемый конечный автомат одной партии
│   │   └── JokerVectorEnv.hpp # OpenMP-пул сред, буфер токенов истории, Oracle и Fork
│   └── joker_c_api.h          # Чистый C ABI для Python / PyTorch / Rust / C#
├── src/                       # Реализация ядра и C API
├── tests/                     # Набор тестов и бенчмарков (pytest)
├── CMakeLists.txt
├── ENGINE.md                  # Подробная документация C API и раскладки тензоров
└── RULES.md                   # Полное описание правил всех режимов игры
```

---

## Сборка

Требования: компилятор с поддержкой **C++23** (`GCC 13+`, `Clang 16+`, `MSVC 2022+`), **CMake 3.20+** и **OpenMP**.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

После сборки в директории `build/` появятся:
* `libjoker_core.so` (или `.dylib` / `.dll`) — разделяемая библиотека с C ABI;
* `joker_cli` — консольная утилита для быстрой проверки и отладки.

---

## Быстрый старт (Python + NumPy / PyTorch)

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("./build/libjoker_core.so")

# Конфигурация: Popular (0), Long (0), Solo (0), Strict (0)
NUM_ENVS = 1024
vec = lib.joker_vec_create(NUM_ENVS, 0, 0, 0, 0, 42, 0)

obs = np.zeros((NUM_ENVS, 384), dtype=np.float32)
masks = np.zeros(NUM_ENVS, dtype=np.uint64)
players = np.zeros(NUM_ENVS, dtype=np.uint8)
rewards = np.zeros((NUM_ENVS, 4), dtype=np.float32)
deal_dones = np.zeros(NUM_ENVS, dtype=np.uint8)
match_dones = np.zeros(NUM_ENVS, dtype=np.uint8)
scores = np.zeros((NUM_ENVS, 4), dtype=np.int32)

# Чтение начальных наблюдений (Zero-Copy)
lib.joker_vec_get_observations(
    vec,
    obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    players.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    None, None, None,
)

# Шаг батча
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

Для запуска тестов необходимо предварительно собрать `libjoker_core.so`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest numpy

# Запуск юнит-тестов правил и тензоров
pytest -v -m "not perf"

# Запуск полного набора вместе с бенчмарками производительности
pytest -v
```

---

## Документация

* **[ENGINE.md](ENGINE.md)** — детальная спецификация пространств действий (`0..45`), раскладки тензоров (`384 float`, `160 float`, `[48, 8] int16`), гибридного поиска (`IS-MCTS`) и всех функций `joker_c_api.h`.
* **[RULES.md](RULES.md)** — исчерпывающее описание правил игры «Джокер» для всех пресетов, длин матчей, парного режима и режимов расчёта пульки.