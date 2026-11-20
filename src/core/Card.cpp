#include "core/Card.hpp"

namespace joker::core {

    std::string Card::ToString() const {
        if (id >= kMaxDeckSize) {
            return "Invalid";
        }
        if (IsJoker()) {
            return id == 36 ? "JK1" : "JK2";
        }
        std::string res{RankToString(rank)};
        res += SuitToString(suit);
        return res;
    }

    std::string CardPlay::ToString() const {
        if (cardId >= kMaxDeckSize) {
            return "InvalidPlay";
        }
        const Card& c = GetCard(cardId);
        if (!c.IsJoker()) {
            return c.ToString();
        }
        std::string res = c.ToString();
        switch (jokerAction) {
        case JokerAction::LeadHighSuit:
            res += "(High:";
            res += SuitToString(requestedSuit);
            res += ")";
            break;
        case JokerAction::LeadLow:
            res += "(Low:";
            res += SuitToString(requestedSuit);
            res += ")";
            break;
        case JokerAction::PlayWin:
            res += "(Win)";
            break;
        case JokerAction::PlayLow:
            res += "(Low)";
            break;
        case JokerAction::None:
            break;
        }
        return res;
    }

} // namespace joker::core