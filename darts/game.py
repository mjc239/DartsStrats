from dataclasses import dataclass
from typing import Union, List


@dataclass
class Player:
    name: str
    score: int


def _convert_throw_string_to_score(throw: str):
    char = throw[0]
    allowed_scores = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34,
        36, 38, 39, 40, 42, 45, 48, 50, 51, 54, 57, 60]
    
    if char.isdigit() and int(throw) in allowed_scores:
        return int(throw)
    elif char == "D":
        return 2 * int(throw[1:])
    elif char == "T":
        return 3 * int(throw[1:])
    elif (throw[:2] == "OB") or (throw[:2] == "SB"):
        return 25
    elif (throw[:2] == "IB") or (throw[:2] == "DB"):
        return 50
    else:
        raise ValueError(f"Throw string {throw} not valid!")


class Leg:
    def __init__(self, players: Union[List[str], int] = 2, start_score: int = 501):
        self.start_score = start_score

        if isinstance(players, int):
            players = [str(i + 1) for i in range(players)]

        self.players = {
            name: Player(name=name, score=self.start_score) for name in players
        }
        self.winner = None
        self.rounds = {name: [self.start_score] for name in self.players}

    def __repr__(self):
        ret = ""
        max_name = max([len(name) for name in self.rounds.keys()])
        for player, scores in self.rounds.items():
            ret += f"{player:>{max_name}}: "
            for score in scores:
                ret += f"{score:>5}"
            ret += f"\n"
        return ret

    def turn(self, player: str, throws: List[str]):
        if self.winner:
            raise AttributeError("Winner already declared!")

        if len(throws) != 3:
            raise ValueError("Expect 3 throws in a turn!")

        current_score = self.players[player].score
        cumulative_score = 0
        for throw in throws:
            throw_value = _convert_throw_string_to_score(throw)
            cumulative_score += throw_value

            if cumulative_score == current_score and throw[0] == "D":
                self.players[player].score = 0
                self.rounds[player].append(0)
                self.winner = player
                return None

        if current_score <= cumulative_score:
            self.rounds[player].append(current_score)
            return None

        self.players[player].score = current_score - cumulative_score
        self.rounds[player].append(current_score - cumulative_score)
