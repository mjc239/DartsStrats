import pytest

from darts import game


def test_leg_defaults():
    leg = game.Leg()
    assert isinstance(leg.players, dict)
    assert list(leg.players.keys()) == ["1", "2"]
    assert isinstance(leg.players["1"], game.Player)
    assert leg.players["1"].score == 501
    assert leg.winner is None
    assert leg.rounds == {"1": [501], "2": [501]}


def test_leg_initialization():
    leg = game.Leg(players=["A", "B", "C"], start_score=1000)

    for name in ["A", "B", "C"]:
        assert name in leg.players
        assert leg.players[name].score == 1000


def test_leg_turn():
    leg = game.Leg()
    leg.turn("1", ["5", "D10", "T20"])

    assert leg.rounds == {"1": [501, 416], "2": [501]}


def test_leg_checkout():
    leg = game.Leg(start_score=20)
    leg.turn("1", ["10", "5", "5"])

    assert leg.rounds == {"1": [20, 20], "2": [20]}

    leg.turn("2", ["5", "5", "D5"])

    assert leg.rounds == {"1": [20, 20], "2": [20, 0]}
    assert leg.winner == "2"

def test_leg_bust():
    leg = game.Leg(start_score=20)
    leg.turn("1", ["10", "10", "10"])

    assert leg.rounds == {"1": [20, 20], "2": [20]}


def test_convert_throw_string_to_score():
    assert game._convert_throw_string_to_score("20") == 20
    assert game._convert_throw_string_to_score("D10") == 20
    assert game._convert_throw_string_to_score("T10") == 30
    assert game._convert_throw_string_to_score("IB") == 50
    assert game._convert_throw_string_to_score("OB") == 25
    with pytest.raises(ValueError):
        game._convert_throw_string_to_score("61")
