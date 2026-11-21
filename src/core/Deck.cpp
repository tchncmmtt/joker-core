#include "core/Deck.hpp"
#include <algorithm>
#include <bit>
#include <random>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace joker::core {

Deck::Deck(const uint64_t seed, const size_t deckSize) noexcept {
    Configure(seed, deckSize);
}

void Deck::InitOrderedDeck() noexcept {
    size_t idx = 0;
    if (deckSize_ == kStandardDeckSize) {
        // 36 карт: без 6♠ (0) и 6♣ (9)
        for (CardId i = 1; i <= 8; ++i) cards_[idx++] = i;
        for (CardId i = 10; i <= 17; ++i) cards_[idx++] = i;
        for (CardId i = 18; i <= 37; ++i) cards_[idx++] = i;
        for (; idx < kMaxDeckSize; ++idx) cards_[idx] = kInvalidCard;
    } else {
        // 38 карт: все от 0 до 37
        for (CardId i = 0; i < kMaxDeckSize; ++i) {
            cards_[idx++] = i;
        }
    }
}

void Deck::Configure(const uint64_t seed, const size_t deckSize) noexcept {
    Reseed(seed);
    SetDeckSize(deckSize);
}

void Deck::Configure256(std::span<const uint64_t, 4> seed, size_t deckSize) noexcept {
    Reseed256(seed);
    SetDeckSize(deckSize);
}

void Deck::SetDeckSize(const size_t deckSize) noexcept {
    deckSize_ = static_cast<uint8_t>(deckSize == kMaxDeckSize ? kMaxDeckSize : kStandardDeckSize);
    if (forceOrderPending_) {
        // Сохраняем принудительно заданный порядок колоды до ближайшей раздачи
        return;
    }
    InitOrderedDeck();
}

void Deck::Reseed(const uint64_t seed) noexcept {
    if (seed == 0) {
        std::random_device rd;
        for (int i = 0; i < 4; ++i) {
            const auto hi = static_cast<uint64_t>(rd());
            const auto lo = static_cast<uint64_t>(rd());
            state_[i] = (hi << 32) ^ lo;
        }
        if ((state_[0] | state_[1] | state_[2] | state_[3]) == 0ULL) {
            state_[0] = 0x9E3779B97F4A7C15ULL;
        }
    } else {
        uint64_t sm = seed;
        for (int i = 0; i < 4; ++i) {
            state_[i] = SplitMix64(sm);
        }
    }
}

void Deck::Reseed256(std::span<const uint64_t, 4> seed) noexcept {
    state_[0] = seed[0];
    state_[1] = seed[1];
    state_[2] = seed[2];
    state_[3] = seed[3];
    if ((state_[0] | state_[1] | state_[2] | state_[3]) == 0ULL) {
        state_[0] = 0x9E3779B97F4A7C15ULL;
        state_[1] = 0xBF58476D1CE4E5B9ULL;
        state_[2] = 0x94D049BB133111EBULL;
        state_[3] = 0x2545F4914F6CDD1DULL;
    }
}

void Deck::ForceDeckOrder(std::span<const CardId, kMaxDeckSize> customCards) noexcept {
    std::copy(customCards.begin(), customCards.end(), cards_.begin());
    forceOrderPending_ = true;
}

void Deck::SetRemainingNinesStock(HandMask remaining24Mask) noexcept {
    size_t idx = 12;
    while (remaining24Mask != 0ULL && idx < kStandardDeckSize) {
        const CardId c = LowestCardInMask(remaining24Mask);
        remaining24Mask &= (remaining24Mask - 1ULL);
        cards_[idx++] = c;
    }
    const uint32_t filled = static_cast<uint32_t>(idx - 12U);
    if (filled > 1U) {
        for (uint32_t i = filled - 1U; i > 0U; --i) {
            const uint32_t j = NextBounded(i + 1U);
            std::swap(cards_[12 + i], cards_[12 + j]);
        }
    }
}

uint64_t Deck::NextRandom() noexcept {
    // Алгоритм xoshiro256** (Blackman & Vigna)
    const uint64_t result = std::rotl(state_[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = state_[1] << 17;

    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];

    state_[2] ^= t;
    state_[3] = std::rotl(state_[3], 45);

    return result;
}

uint32_t Deck::NextBounded(const uint32_t range) noexcept {
    if (range <= 1U) return 0U;
    const auto r = static_cast<uint64_t>(range);
#if defined(__SIZEOF_INT128__)
    __extension__ using u128 = unsigned __int128;
    uint64_t x = NextRandom();
    u128 m = static_cast<u128>(x) * static_cast<u128>(r);
    auto l = static_cast<uint64_t>(m);
    if (l < r) {
        const uint64_t threshold = (0ULL - r) % r;
        while (l < threshold) {
            x = NextRandom();
            m = static_cast<u128>(x) * static_cast<u128>(r);
            l = static_cast<uint64_t>(m);
        }
    }
    return static_cast<uint32_t>(m >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
    uint64_t x = NextRandom();
    uint64_t hi = 0;
    uint64_t l = _umul128(x, r, &hi);
    if (l < r) {
        const uint64_t threshold = (0ULL - r) % r;
        while (l < threshold) {
            x = NextRandom();
            l = _umul128(x, r, &hi);
        }
    }
    return static_cast<uint32_t>(hi);
#else
    const uint64_t threshold = (0ULL - r) % r;
    for (;;) {
        const uint64_t x = NextRandom();
        if (x >= threshold) {
            return static_cast<uint32_t>(x % r);
        }
    }
#endif
}

void Deck::Shuffle() noexcept {
    if (forceOrderPending_) {
        forceOrderPending_ = false;
        return;
    }
    // Тасование Фишера-Йетса с алгоритмом Лемира без modulo bias
    for (uint32_t i = static_cast<uint32_t>(deckSize_) - 1U; i > 0U; --i) {
        const uint32_t j = NextBounded(i + 1U);
        std::swap(cards_[i], cards_[j]);
    }
}

DealResult Deck::DealStandard(int cardsPerPlayer) noexcept {
    Shuffle();
    DealResult res{};

    size_t cardIdx = 0;
    for (int p = 0; p < kNumPlayers; ++p) {
        for (int c = 0; c < cardsPerPlayer; ++c) {
            res.hands[p] |= (1ULL << cards_[cardIdx++]);
        }
    }

    if (cardIdx < deckSize_) {
        res.upcard = cards_[cardIdx];
        const Card& up = GetCard(res.upcard);
        res.trumpSuit = up.IsJoker() ? Suit::None : up.suit;
    } else {
        res.upcard = kInvalidCard;
        res.trumpSuit = Suit::None;
    }

    return res;
}

std::array<HandMask, kNumPlayers> Deck::DealFirstThree() noexcept {
    Shuffle();
    std::array<HandMask, kNumPlayers> hands{};
    size_t cardIdx = 0;
    for (int p = 0; p < kNumPlayers; ++p) {
        for (int c = 0; c < 3; ++c) {
            hands[p] |= (1ULL << cards_[cardIdx++]);
        }
    }
    return hands;
}

void Deck::DealRemainingSix(std::span<HandMask, kNumPlayers> hands) const noexcept {
    size_t cardIdx = 12;
    for (int p = 0; p < kNumPlayers; ++p) {
        for (int c = 0; c < 6; ++c) {
            hands[p] |= (1ULL << cards_[cardIdx++]);
        }
    }
}

} // namespace joker::core