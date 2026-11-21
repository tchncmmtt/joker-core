#pragma once

#include "core/Action.hpp"
#include "engine/JokerEngine.hpp"
#include <cstdint>

namespace joker::engine {

    class HeuristicBot {
    public:
        // Быстрая безаллокационная 3-фазная эвристика (~300-500 нс на ход)
        [[nodiscard]] static core::ActionId SelectAction(
            const JokerEngine& env,
            uint64_t& rngState) noexcept;
    };

} // namespace joker::engine