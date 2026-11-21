#include "engine/Determinizer.hpp"
#include "core/Rules.hpp"
#include <array>
#include <utility>

namespace joker::engine {

using namespace joker::core;

void DeterminizeState(
    GameState& state,
    Deck& deck,
    PlayerId observer,
    const uint64_t rngSeed) noexcept {
    if (state.phase == Phase::MatchFinished || state.IsWaitingForDealInput()) {
        return;
    }
    observer = static_cast<PlayerId>(observer % kNumPlayers);

    deck.Reseed(rngSeed);

    const HandMask fullDeck = rules::GetFullDeckMask(state.rules);
    const HandMask upcardMask = (state.upcard != kInvalidCard && state.upcard < kMaxDeckSize)
        ? (1ULL << state.upcard)
        : 0ULL;
    const HandMask knownMask = state.hands[observer] | state.playedCardsHistory | upcardMask;
    const HandMask unknownMask = fullDeck & ~knownMask;

    std::array<CardId, kMaxDeckSize> pool{};
    uint32_t poolSize = 0;
    HandMask tmp = unknownMask;
    while (tmp != 0ULL) {
        const CardId cid = LowestCardInMask(tmp);
        tmp &= (tmp - 1ULL);
        pool[poolSize++] = cid;
    }

    if (poolSize == 0) return;

    std::array<PlayerId, 3> opps{};
    std::array<uint32_t, 3> need{};
    std::array<HandMask, 3> forbidden{};
    int oppIdx = 0;
    uint32_t totalNeeded = 0;

    for (PlayerId p = 0; p < kNumPlayers; ++p) {
        if (p == observer) continue;
        opps[oppIdx] = p;
        need[oppIdx] = state.handCounts[p];
        totalNeeded += need[oppIdx];
        forbidden[oppIdx] = state.impossibleCards[p];
        oppIdx++;
    }

    if (totalNeeded > poolSize) return;

    for (uint32_t i = poolSize - 1U; i > 0U; --i) {
        const uint32_t j = deck.NextBounded(i + 1U);
        std::swap(pool[i], pool[j]);
    }

    const std::array<uint32_t, 4> segStart{0, need[0], need[0] + need[1], totalNeeded};
    const std::array<uint32_t, 4> segEnd{need[0], need[0] + need[1], totalNeeded, poolSize};

    auto isCardAllowedForBucket = [&](const CardId c, const int bucket) noexcept -> bool {
        if (bucket >= 3) return true;
        return ((1ULL << c) & forbidden[bucket]) == 0ULL;
    };

    // Проход 1: быстрый жадный обмен вперёд
    for (int b = 0; b < 3; ++b) {
        for (uint32_t pos = segStart[b]; pos < segEnd[b]; ++pos) {
            if (!isCardAllowedForBucket(pool[pos], b)) {
                for (uint32_t cand = pos + 1U; cand < poolSize; ++cand) {
                    if (isCardAllowedForBucket(pool[cand], b)) {
                        std::swap(pool[pos], pool[cand]);
                        break;
                    }
                }
            }
        }
    }

    // Проход 2: парный обмен между любыми двумя корзинами
    for (int b = 0; b < 3; ++b) {
        for (uint32_t pos = segStart[b]; pos < segEnd[b]; ++pos) {
            const CardId badCard = pool[pos];
            if (isCardAllowedForBucket(badCard, b)) continue;

            bool repaired = false;
            for (int b2 = 3; b2 >= 0 && !repaired; --b2) {
                if (b2 == b) continue;
                for (uint32_t pos2 = segStart[b2]; pos2 < segEnd[b2]; ++pos2) {
                    const CardId otherCard = pool[pos2];
                    if (isCardAllowedForBucket(badCard, b2) && isCardAllowedForBucket(otherCard, b)) {
                        std::swap(pool[pos], pool[pos2]);
                        repaired = true;
                        break;
                    }
                }
            }
        }
    }

    // Проход 3: точный BFS по дополняющим циклам среди 4 корзин (0..3) на стеке, без аллокаций
    for (int b = 0; b < 3; ++b) {
        for (uint32_t pos = segStart[b]; pos < segEnd[b]; ++pos) {
            if (isCardAllowedForBucket(pool[pos], b)) continue;

            std::array<int, 4> parentBucket{-1, -1, -1, -1};
            std::array<uint32_t, 4> edgePos{0, 0, 0, 0};
            std::array<int, 4> queue{};
            int qHead = 0;
            int qTail = 0;

            for (int nextB = 0; nextB < 4; ++nextB) {
                if (nextB == b || segStart[nextB] == segEnd[nextB]) continue;
                if (isCardAllowedForBucket(pool[pos], nextB)) {
                    parentBucket[nextB] = b;
                    edgePos[nextB] = pos;
                    queue[qTail++] = nextB;
                }
            }

            int foundEndBucket = -1;
            uint32_t closingPos = 0;

            while (qHead < qTail && foundEndBucket == -1) {
                const int curB = queue[qHead++];
                for (uint32_t p2 = segStart[curB]; p2 < segEnd[curB]; ++p2) {
                    if (isCardAllowedForBucket(pool[p2], b)) {
                        foundEndBucket = curB;
                        closingPos = p2;
                        break;
                    }
                }
                if (foundEndBucket != -1) break;

                for (int nextB = 0; nextB < 4; ++nextB) {
                    if (nextB == b || parentBucket[nextB] != -1 || segStart[nextB] == segEnd[nextB]) {
                        continue;
                    }
                    for (uint32_t p2 = segStart[curB]; p2 < segEnd[curB]; ++p2) {
                        if (isCardAllowedForBucket(pool[p2], nextB)) {
                            parentBucket[nextB] = curB;
                            edgePos[nextB] = p2;
                            queue[qTail++] = nextB;
                            break;
                        }
                    }
                }
            }

            if (foundEndBucket != -1) {
                std::array<uint32_t, 4> cyclePos{};
                int len = 0;
                cyclePos[len++] = closingPos;
                int cur = foundEndBucket;
                while (cur != b && len < 4) {
                    cyclePos[len++] = edgePos[cur];
                    cur = parentBucket[cur];
                }
                const CardId lastCard = pool[cyclePos[0]];
                for (int k = 0; k + 1 < len; ++k) {
                    pool[cyclePos[k]] = pool[cyclePos[k + 1]];
                }
                pool[cyclePos[len - 1]] = lastCard;
            }
        }
    }

    for (int b = 0; b < 3; ++b) {
        HandMask newHand = 0ULL;
        for (uint32_t pos = segStart[b]; pos < segEnd[b]; ++pos) {
            newHand |= (1ULL << pool[pos]);
        }
        state.hands[opps[b]] = newHand;
    }

    state.knownHandsMask = 0x0F;

    if (state.phase == Phase::ChoosingTrump) {
        HandMask remaining24Mask = 0ULL;
        for (uint32_t pos = segStart[3]; pos < segEnd[3]; ++pos) {
            remaining24Mask |= (1ULL << pool[pos]);
        }
        deck.SetRemainingNinesStock(remaining24Mask);
    }
}

} // namespace joker::engine