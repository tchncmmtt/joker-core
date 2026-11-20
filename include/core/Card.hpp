#pragma once

#include "core/Types.hpp"
#include <array>
#include <bit>
#include <string>
#include <utility>

namespace joker::core {

struct Card {
    CardId id{kInvalidCard};
    Suit suit{Suit::None};
    Rank rank{Rank::Joker};

    [[nodiscard]] constexpr bool IsJoker() const noexcept {
        return id == 36 || id == 37;
    }

    [[nodiscard]] constexpr HandMask Bit() const noexcept {
        return 1ULL << id;
    }

    [[nodiscard]] std::string ToString() const;
};

struct CardPlay {
    CardId cardId{kInvalidCard};
    JokerAction jokerAction{JokerAction::None};
    Suit requestedSuit{Suit::None};

    constexpr bool operator==(const CardPlay& other) const noexcept = default;
    [[nodiscard]] std::string ToString() const;
};

struct TrickCard {
    PlayerId player{0};
    CardPlay play{};
};

inline constexpr std::array<Card, kMaxDeckSize> kAllCards = {{
    // Spades (0..8)
    {0, Suit::Spades, Rank::Six},   {1, Suit::Spades, Rank::Seven},
    {2, Suit::Spades, Rank::Eight}, {3, Suit::Spades, Rank::Nine},
    {4, Suit::Spades, Rank::Ten},   {5, Suit::Spades, Rank::Jack},
    {6, Suit::Spades, Rank::Queen}, {7, Suit::Spades, Rank::King},
    {8, Suit::Spades, Rank::Ace},
    // Clubs (9..17)
    {9, Suit::Clubs, Rank::Six},    {10, Suit::Clubs, Rank::Seven},
    {11, Suit::Clubs, Rank::Eight}, {12, Suit::Clubs, Rank::Nine},
    {13, Suit::Clubs, Rank::Ten},   {14, Suit::Clubs, Rank::Jack},
    {15, Suit::Clubs, Rank::Queen}, {16, Suit::Clubs, Rank::King},
    {17, Suit::Clubs, Rank::Ace},
    // Diamonds (18..26)
    {18, Suit::Diamonds, Rank::Six},   {19, Suit::Diamonds, Rank::Seven},
    {20, Suit::Diamonds, Rank::Eight}, {21, Suit::Diamonds, Rank::Nine},
    {22, Suit::Diamonds, Rank::Ten},   {23, Suit::Diamonds, Rank::Jack},
    {24, Suit::Diamonds, Rank::Queen}, {25, Suit::Diamonds, Rank::King},
    {26, Suit::Diamonds, Rank::Ace},
    // Hearts (27..35)
    {27, Suit::Hearts, Rank::Six},   {28, Suit::Hearts, Rank::Seven},
    {29, Suit::Hearts, Rank::Eight}, {30, Suit::Hearts, Rank::Nine},
    {31, Suit::Hearts, Rank::Ten},   {32, Suit::Hearts, Rank::Jack},
    {33, Suit::Hearts, Rank::Queen}, {34, Suit::Hearts, Rank::King},
    {35, Suit::Hearts, Rank::Ace},
    // Jokers (36..37)
    {36, Suit::None, Rank::Joker},
    {37, Suit::None, Rank::Joker}
}};

inline constexpr std::array<HandMask, 5> kSuitMasks = {
    0x00000001FFULL, // Spades   (bits 0..8)
    0x000003FE00ULL, // Clubs    (bits 9..17)
    0x0007FC0000ULL, // Diamonds (bits 18..26)
    0x0FF8000000ULL, // Hearts   (bits 27..35)
    0x0000000000ULL  // None
};

inline constexpr HandMask kJokerMask = (1ULL << 36) | (1ULL << 37);
inline constexpr HandMask kClassicDeckMask = (1ULL << 38) - 1ULL;
inline constexpr HandMask kStandardDeckMask = kClassicDeckMask & ~((1ULL << 0) | (1ULL << 9));

constexpr const Card& GetCard(const CardId id) noexcept {
    [[assume(id < kMaxDeckSize)]];
    return kAllCards[id];
}

constexpr HandMask GetSuitMask(const Suit s) noexcept {
    const auto idx = std::to_underlying(s);
    [[assume(idx < kSuitMasks.size())]];
    return kSuitMasks[idx];
}

constexpr CardId HighestCardInMask(const HandMask mask) noexcept {
    [[assume(mask != 0)]];
    return static_cast<CardId>(63 - std::countl_zero(mask));
}

constexpr CardId LowestCardInMask(const HandMask mask) noexcept {
    [[assume(mask != 0)]];
    return static_cast<CardId>(std::countr_zero(mask));
}

} // namespace joker::core