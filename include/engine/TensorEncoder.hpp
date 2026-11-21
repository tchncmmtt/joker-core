#pragma once

#include "core/GameState.hpp"
#include <cstddef>

namespace joker::engine {

    inline constexpr size_t kOracleTensorSize = 160;

    void WriteOracleTensor(
        const core::GameState& state,
        core::PlayerId observer,
        float* outOracle) noexcept;

} // namespace joker::engine