#pragma once

#include "core/Deck.hpp"
#include "core/GameState.hpp"
#include <cstdint>

namespace joker::engine {

    void DeterminizeState(
        core::GameState& state,
        core::Deck& deck,
        core::PlayerId observer,
        uint64_t rngSeed) noexcept;

} // namespace joker::engine