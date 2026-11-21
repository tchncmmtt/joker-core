#include "engine/ArenaStats.hpp"
#include "core/GameState.hpp"
#include "core/Rules.hpp"
#include <algorithm>
#include <limits>
#include <utility>

namespace joker::engine {

using namespace joker::core;

void SlotStats::Reset() noexcept {
    for (int p = 0; p < kNumPlayers; ++p) {
        players[p] = JokerPlayerStats{};
        players[p].max_match_score = std::numeric_limits<int32_t>::min();
        players[p].min_match_score = std::numeric_limits<int32_t>::max();
        players[p].max_deal_score = std::numeric_limits<int32_t>::min();
        players[p].min_deal_score = std::numeric_limits<int32_t>::max();
    }
    matchBonusesCount.fill(0U);
}

void RecordSlotStepStats(
    SlotStats& ss,
    const RulePreset preset,
    const MatchMode mode,
    const TeamMode teamMode,
    const Phase prevPhase,
    const PlayerId actor,
    const ActionId action,
    const int prevDeal,
    const int prevCardsInDeal,
    const std::array<int8_t, kNumPlayers>& prevBids,
    const std::array<int8_t, kNumPlayers>& finalDealTricks,
    const std::array<int32_t, kNumPlayers>& prevScores,
    const std::array<int32_t, kNumPlayers>& afterScores,
    const std::array<bool, kNumPlayers>& prevPulkaClean,
    const bool dealDone,
    const bool matchDone) noexcept {

    if (prevPhase == Phase::ChoosingTrump) {
        ss.players[actor].nines_trump_choices++;
        if (action == std::to_underlying(Suit::None)) {
            ss.players[actor].nines_no_trump_choices++;
        }
    } else if (prevPhase == Phase::TrickPlay && action >= 36U) {
        if (action == kActionJokerPlayWin) {
            ss.players[actor].joker_play_win++;
        } else if (action == kActionJokerPlayLow) {
            ss.players[actor].joker_play_low++;
        } else if (action >= kActionJokerLeadHighBase && action < kActionJokerLeadHighBase + 4U) {
            ss.players[actor].joker_lead_high++;
        } else if (action >= kActionJokerLeadLowBase && action < kActionJokerLeadLowBase + 4U) {
            ss.players[actor].joker_lead_low++;
        }
    }

    if (dealDone) {
        const auto& sched = GetScheduleEntry(mode, prevDeal);
        for (int p = 0; p < kNumPlayers; ++p) {
            JokerPlayerStats& st = ss.players[p];
            const int b = prevBids[p];
            const int t = finalDealTricks[p];
            const int32_t dealScore = static_cast<int32_t>(
                rules::CalculateDealScore(preset, b, t, prevCardsInDeal));

            st.deals_played++;
            st.max_deal_score = std::max(st.max_deal_score, dealScore);
            st.min_deal_score = std::min(st.min_deal_score, dealScore);

            if (t == b) {
                st.exact_bids++;
            } else if (t > b) {
                st.overbids++;
            } else {
                st.underbids++;
                if (b > 0 && t == 0) {
                    st.shangas++;
                }
            }

            if (b >= 4) {
                st.large_bids_count++;
                if (t == b) {
                    st.large_bids_exact++;
                }
            }

            if (prevCardsInDeal == 9) {
                st.deals_9_count++;
                st.bids_sum_9 += static_cast<uint64_t>(std::max(0, b));
                st.tricks_sum_9 += static_cast<uint64_t>(std::max(0, t));
            } else {
                st.deals_1_8_count++;
                st.bids_sum_1_8 += static_cast<uint64_t>(std::max(0, b));
                st.tricks_sum_1_8 += static_cast<uint64_t>(std::max(0, t));
            }

            if (sched.isPulkaEnd) {
                st.pulkas_played++;
                if (prevPulkaClean[p] && (b == t)) {
                    st.pulkas_clean++;
                }
                const int32_t pulkaDelta = (afterScores[p] - prevScores[p]) - dealScore;
                if (pulkaDelta > 0) {
                    st.pulkas_bonus_received++;
                    ss.matchBonusesCount[p]++;
                } else if (pulkaDelta < 0) {
                    st.pulkas_penalty_received++;
                }
            }
        }
    }

    if (matchDone) {
        if (teamMode == TeamMode::Pairs) {
            const int32_t teamA = afterScores[0] + afterScores[2];
            const int32_t teamB = afterScores[1] + afterScores[3];
            for (int p = 0; p < kNumPlayers; ++p) {
                const bool isA = (p == 0 || p == 2);
                const int32_t myTeam = isA ? teamA : teamB;
                const int32_t oppTeam = isA ? teamB : teamA;
                if (myTeam > oppTeam) {
                    ss.players[p].place_counts[0]++;
                    ss.players[p].place_points += 1.0;
                } else if (myTeam < oppTeam) {
                    ss.players[p].place_counts[1]++;
                } else {
                    ss.players[p].place_counts[0]++;
                    ss.players[p].place_points += 0.5;
                }
            }
        } else {
            for (int p = 0; p < kNumPlayers; ++p) {
                int strictlyBetter = 0;
                int equalCount = 0;
                for (int q = 0; q < kNumPlayers; ++q) {
                    if (afterScores[q] > afterScores[p]) {
                        ++strictlyBetter;
                    } else if (afterScores[q] == afterScores[p]) {
                        ++equalCount;
                    }
                }
                ss.players[p].place_counts[static_cast<size_t>(strictlyBetter)]++;
                if (strictlyBetter == 0) {
                    ss.players[p].place_points += 1.0 / static_cast<double>(equalCount);
                }
            }
        }

        for (int p = 0; p < kNumPlayers; ++p) {
            JokerPlayerStats& st = ss.players[p];
            const int32_t sc = afterScores[p];
            st.matches_played++;
            st.total_match_score_sum += static_cast<int64_t>(sc);
            st.total_match_score_sq_sum += static_cast<double>(sc) * static_cast<double>(sc);
            st.max_match_score = std::max(st.max_match_score, sc);
            st.min_match_score = std::min(st.min_match_score, sc);
            if (ss.matchBonusesCount[p] >= 2U) {
                st.matches_multi_bonus++;
            }
            ss.matchBonusesCount[p] = 0U;
        }
    }
}

void AggregateSlotStats(
    std::span<const SlotStats> slotStats,
    std::span<JokerPlayerStats, kNumPlayers> outStats) noexcept {
    for (int p = 0; p < kNumPlayers; ++p) {
        outStats[p] = JokerPlayerStats{};
        outStats[p].max_match_score = std::numeric_limits<int32_t>::min();
        outStats[p].min_match_score = std::numeric_limits<int32_t>::max();
        outStats[p].max_deal_score = std::numeric_limits<int32_t>::min();
        outStats[p].min_deal_score = std::numeric_limits<int32_t>::max();
    }

    for (const auto& ss : slotStats) {
        for (int p = 0; p < kNumPlayers; ++p) {
            const JokerPlayerStats& src = ss.players[p];
            JokerPlayerStats& dst = outStats[p];

            dst.matches_played += src.matches_played;
            for (int r = 0; r < 4; ++r) {
                dst.place_counts[r] += src.place_counts[r];
            }
            dst.place_points += src.place_points;
            dst.total_match_score_sum += src.total_match_score_sum;
            dst.total_match_score_sq_sum += src.total_match_score_sq_sum;
            if (src.matches_played > 0) {
                dst.max_match_score = std::max(dst.max_match_score, src.max_match_score);
                dst.min_match_score = std::min(dst.min_match_score, src.min_match_score);
            }

            dst.deals_played += src.deals_played;
            if (src.deals_played > 0) {
                dst.max_deal_score = std::max(dst.max_deal_score, src.max_deal_score);
                dst.min_deal_score = std::min(dst.min_deal_score, src.min_deal_score);
            }
            dst.exact_bids += src.exact_bids;
            dst.overbids += src.overbids;
            dst.underbids += src.underbids;
            dst.shangas += src.shangas;
            dst.large_bids_count += src.large_bids_count;
            dst.large_bids_exact += src.large_bids_exact;

            dst.deals_1_8_count += src.deals_1_8_count;
            dst.bids_sum_1_8 += src.bids_sum_1_8;
            dst.tricks_sum_1_8 += src.tricks_sum_1_8;
            dst.deals_9_count += src.deals_9_count;
            dst.bids_sum_9 += src.bids_sum_9;
            dst.tricks_sum_9 += src.tricks_sum_9;

            dst.nines_trump_choices += src.nines_trump_choices;
            dst.nines_no_trump_choices += src.nines_no_trump_choices;

            dst.joker_lead_high += src.joker_lead_high;
            dst.joker_lead_low += src.joker_lead_low;
            dst.joker_play_win += src.joker_play_win;
            dst.joker_play_low += src.joker_play_low;

            dst.pulkas_played += src.pulkas_played;
            dst.pulkas_clean += src.pulkas_clean;
            dst.pulkas_bonus_received += src.pulkas_bonus_received;
            dst.pulkas_penalty_received += src.pulkas_penalty_received;
            dst.matches_multi_bonus += src.matches_multi_bonus;
        }
    }

    for (int p = 0; p < kNumPlayers; ++p) {
        if (outStats[p].matches_played == 0) {
            outStats[p].max_match_score = 0;
            outStats[p].min_match_score = 0;
        }
        if (outStats[p].deals_played == 0) {
            outStats[p].max_deal_score = 0;
            outStats[p].min_deal_score = 0;
        }
    }
}

} // namespace joker::engine