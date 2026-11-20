#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

namespace joker::core {

constexpr int kNumPlayers = 4;
constexpr int kLongDeals = 24;
constexpr int kLongPulkasCount = 4;
constexpr int kShortDeals = 8;
constexpr int kShortPulkasCount = 2;
constexpr size_t kMaxDeckSize = 38;
constexpr size_t kStandardDeckSize = 36;

using PlayerId = uint8_t;
using CardId = uint8_t;
using HandMask = uint64_t;

constexpr CardId kInvalidCard = 255;

enum class MatchMode : uint8_t {
    Long  = 0,
    Short = 1
};

enum class RulePreset : uint8_t {
    Popular = 0,
    Minus   = 1,
    Classic = 2
};

enum class TeamMode : uint8_t {
    Solo  = 0,
    Pairs = 1
};

enum class PulkaMode : uint8_t {
    Strict = 0,
    Soft   = 1
};

enum class Suit : uint8_t {
    Spades   = 0,
    Clubs    = 1,
    Diamonds = 2,
    Hearts   = 3,
    None     = 4
};

enum class Rank : uint8_t {
    Six   = 0,
    Seven = 1,
    Eight = 2,
    Nine  = 3,
    Ten   = 4,
    Jack  = 5,
    Queen = 6,
    King  = 7,
    Ace   = 8,
    Joker = 9
};

enum class JokerAction : uint8_t {
    None         = 0,
    LeadHighSuit = 1,
    LeadLow      = 2,
    PlayWin      = 3,
    PlayLow      = 4
};

constexpr std::string_view SuitToString(const Suit s) noexcept {
    switch (s) {
        case Suit::Spades:   return "♠";
        case Suit::Clubs:    return "♣";
        case Suit::Diamonds: return "♦";
        case Suit::Hearts:   return "♥";
        case Suit::None:     return "Б/К";
    }
    std::unreachable();
}

constexpr std::string_view RankToString(const Rank r) noexcept {
    switch (r) {
        case Rank::Six:   return "6";
        case Rank::Seven: return "7";
        case Rank::Eight: return "8";
        case Rank::Nine:  return "9";
        case Rank::Ten:   return "10";
        case Rank::Jack:  return "J";
        case Rank::Queen: return "Q";
        case Rank::King:  return "K";
        case Rank::Ace:   return "A";
        case Rank::Joker: return "JK";
    }
    std::unreachable();
}

constexpr std::string_view MatchModeToString(const MatchMode m) noexcept {
    switch (m) {
        case MatchMode::Long:  return "long";
        case MatchMode::Short: return "short";
    }
    std::unreachable();
}

constexpr std::string_view RulePresetToString(const RulePreset r) noexcept {
    switch (r) {
        case RulePreset::Popular: return "popular";
        case RulePreset::Minus:   return "minus";
        case RulePreset::Classic: return "classic";
    }
    std::unreachable();
}

constexpr std::string_view TeamModeToString(const TeamMode t) noexcept {
    switch (t) {
        case TeamMode::Solo:  return "solo";
        case TeamMode::Pairs: return "pairs";
    }
    std::unreachable();
}

constexpr std::string_view PulkaModeToString(const PulkaMode p) noexcept {
    switch (p) {
        case PulkaMode::Strict: return "strict";
        case PulkaMode::Soft:   return "soft";
    }
    std::unreachable();
}

} // namespace joker::core