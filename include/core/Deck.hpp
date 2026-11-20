#pragma once

#include "core/Card.hpp"
#include "core/Types.hpp"
#include <array>
#include <cstdint>
#include <span>

namespace joker::core {

    struct DealResult {
        std::array<HandMask, kNumPlayers> hands{};
        CardId upcard{kInvalidCard};
        Suit trumpSuit{Suit::None};
    };

    class Deck {
    public:
        explicit Deck(uint64_t seed = 42, size_t deckSize = kStandardDeckSize) noexcept;

        void Configure(uint64_t seed, size_t deckSize) noexcept;
        void Configure256(std::span<const uint64_t, 4> seed, size_t deckSize) noexcept;
        void SetDeckSize(size_t deckSize) noexcept;

        void Reseed(uint64_t seed) noexcept;
        void Reseed256(std::span<const uint64_t, 4> seed) noexcept;

        void ForceDeckOrder(std::span<const CardId, kMaxDeckSize> customCards) noexcept;
        void SetRemainingNinesStock(HandMask remaining24Mask) noexcept;

        [[nodiscard]] DealResult DealStandard(int cardsPerPlayer) noexcept;
        [[nodiscard]] std::array<HandMask, kNumPlayers> DealFirstThree() noexcept;
        void DealRemainingSix(std::span<HandMask, kNumPlayers> hands) const noexcept;

        [[nodiscard]] uint64_t NextRandom() noexcept;
        [[nodiscard]] uint32_t NextBounded(uint32_t range) noexcept;

        [[nodiscard]] const std::array<CardId, kMaxDeckSize>& GetCards() const noexcept { return cards_; }
        [[nodiscard]] uint8_t GetDeckSize() const noexcept { return deckSize_; }

        static constexpr uint64_t SplitMix64(uint64_t& x) noexcept {
            uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        }

    private:
        void Shuffle() noexcept;

        std::array<uint64_t, 4> state_{
            0x123456789ABCDEF0ULL,
            0xFEDCBA9876543210ULL,
            0x0F1E2D3C4B5A6978ULL,
            0x8796A5B4C3D2E1F0ULL
        };
        std::array<CardId, kMaxDeckSize> cards_{};
        uint8_t deckSize_{kStandardDeckSize};
        bool forceOrderPending_{false};
    };

} // namespace joker::core