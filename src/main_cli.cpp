#include "engine/JokerEngine.hpp"
#include "engine/JokerVectorEnv.hpp"
#include <chrono>
#include <iostream>

int main() {
    using namespace joker::core;
    using namespace joker::engine;

    std::cout << "Joker Core Engine CLI\n";
    std::cout << "  sizeof(GameState)   = " << sizeof(GameState) << " bytes\n";
    std::cout << "  sizeof(JokerEngine) = " << sizeof(JokerEngine) << " bytes\n";
    std::cout << "  trivially_copyable  = " << std::boolalpha
              << std::is_trivially_copyable_v<JokerEngine> << "\n";

    JokerEngine env(RulePreset::Popular, 42, MatchMode::Long, TeamMode::Solo, PulkaMode::Strict);
    uint64_t rng = 123456789ULL;
    size_t steps = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (!env.IsDone()) {
        const ActionId act = JokerVectorEnv::SelectHeuristicAction(env, rng);
        env.Step(act);
        ++steps;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const auto& sc = env.GetState().totalScores;
    std::cout << "  Sample heuristic match: " << steps << " steps in " << us << " us | scores = ["
              << sc[0] << ", " << sc[1] << ", " << sc[2] << ", " << sc[3] << "]\n";
    return 0;
}