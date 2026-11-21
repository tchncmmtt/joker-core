#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#  define JOKER_UNREACHABLE() ::std::unreachable()
#elif defined(_MSC_VER)
#  define JOKER_UNREACHABLE() __assume(false)
#elif defined(__GNUC__) || defined(__clang__)
#  define JOKER_UNREACHABLE() __builtin_unreachable()
#else
#  define JOKER_UNREACHABLE() ((void)0)
#endif

#if defined(__has_cpp_attribute)
#  if __has_cpp_attribute(assume) >= 202207L
#    define JOKER_ASSUME(expr) [[assume(expr)]]
#  endif
#endif
#ifndef JOKER_ASSUME
#  if defined(_MSC_VER)
#    define JOKER_ASSUME(expr) __assume(expr)
#  elif defined(__clang__)
#    define JOKER_ASSUME(expr) __builtin_assume(expr)
#  else
#    define JOKER_ASSUME(expr) ((void)0)
#  endif
#endif

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
    Long = 0,
    Short = 1
};

enum class RulePreset : uint8_t {
    Popular = 0,
    Minus = 1,
    Classic = 2
};

enum class TeamMode : uint8_t {
    Solo = 0,
    Pairs = 1
};

enum class PulkaMode : uint8_t {
    Strict = 0,
    Soft = 1
};

enum class Suit : uint8_t {
    Spades = 0,
    Clubs = 1,
    Diamonds = 2,
    Hearts = 3,
    None = 4
};

enum class Rank : uint8_t {
    Six = 0,
    Seven = 1,
    Eight = 2,
    Nine = 3,
    Ten = 4,
    Jack = 5,
    Queen = 6,
    King = 7,
    Ace = 8,
    Joker = 9
};

enum class JokerAction : uint8_t {
    None = 0,
    LeadHighSuit = 1,
    LeadLow = 2,
    PlayWin = 3,
    PlayLow = 4
};

enum class StepError : uint8_t {
    Ok = 0,
    NullHandle = 1,
    MatchFinished = 2,
    WaitingForDealInput = 3,
    InvalidActionId = 4,
    ForbiddenDealerBid = 5,
    CardNotInHand = 6,
    CardAlreadyPlayed = 7,
    CardInOtherKnownHand = 8,
    CardIsUpcard = 9,
    MustFollowSuit = 10,
    MustPlayTrump = 11,
    MustPlayHighest = 12,
    ViolatesKnownVoidOrBound = 13,
    NoJokerAvailable = 14
};

constexpr std::string_view SuitToString(const Suit s) noexcept {
    switch (s) {
        case Suit::Spades:   return "♠";
        case Suit::Clubs:    return "♣";
        case Suit::Diamonds: return "♦";
        case Suit::Hearts:   return "♥";
        case Suit::None:     return "Б/К";
    }
    JOKER_UNREACHABLE();
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
    JOKER_UNREACHABLE();
}

constexpr std::string_view MatchModeToString(const MatchMode m) noexcept {
    switch (m) {
        case MatchMode::Long:  return "long";
        case MatchMode::Short: return "short";
    }
    JOKER_UNREACHABLE();
}

constexpr std::string_view RulePresetToString(const RulePreset r) noexcept {
    switch (r) {
        case RulePreset::Popular: return "popular";
        case RulePreset::Minus:   return "minus";
        case RulePreset::Classic: return "classic";
    }
    JOKER_UNREACHABLE();
}

constexpr std::string_view TeamModeToString(const TeamMode t) noexcept {
    switch (t) {
        case TeamMode::Solo:  return "solo";
        case TeamMode::Pairs: return "pairs";
    }
    JOKER_UNREACHABLE();
}

constexpr std::string_view PulkaModeToString(const PulkaMode p) noexcept {
    switch (p) {
        case PulkaMode::Strict: return "strict";
        case PulkaMode::Soft:   return "soft";
    }
    JOKER_UNREACHABLE();
}

} // namespace joker::core