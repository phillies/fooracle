from typing import Annotated, TypeAlias

import numpy as np
import pandas as pd
import typer

from fooracle.models.base import ConstantFooracle, Fooracle

GameResult: TypeAlias = tuple[int, int]
GameTeams: TypeAlias = tuple[str, str]


def get_winner(game: GameTeams, result: GameResult) -> str:
    """
    Determines the winner of a game based on the result.

    :param game: A tuple containing the names of the two players.
    :type game: GameTeams
    :param result: A tuple containing the scores of the two players.
    :type result: GameResult
    :return: The name of the winner.
    :rtype: str
    """
    if result[0] > result[1]:
        winner = game[0]
    elif result[0] < result[1]:
        winner = game[1]
    else:
        # if there is a draw we toss a coin
        winner = game[np.random.randint(2)]
    return winner


def calculate_league_standing(
    teams: list[str], games: list[GameTeams], results: list[GameResult]
) -> pd.DataFrame:
    standings = pd.DataFrame(
        index=teams, columns=["points", "goals_scored", "goals_taken"]
    ).fillna(0)
    for game, result in zip(games, results, strict=True):
        if result[0] == result[1]:
            standings.loc[game[0]].points += 1
            standings.loc[game[1]].points += 1
        elif result[0] > result[1]:
            standings.loc[game[0]].points += 3
        else:
            standings.loc[game[1]].points += 3

        standings.loc[game[0]].goals_scored += result[0]
        standings.loc[game[0]].goals_taken += result[1]

        standings.loc[game[1]].goals_scored += result[1]
        standings.loc[game[1]].goals_taken += result[0]
    standings["goal_difference"] = standings["goals_scored"] - standings["goals_taken"]
    standings = standings.sort_values(
        by=["points", "goal_difference", "goals_scored"], ascending=False
    )
    return standings


def foretell_games(
    fooracle: Fooracle,
    games: list[GameTeams],
    draw_allowed: bool = True,
    host: str | None = None,
) -> list[GameResult]:
    """Predicts all results in the list games. Each entry must be a list of 2 countries."""
    results = []
    for game in games:
        game_result = fooracle.foretell(
            game[0], game[1], draw_allowed=draw_allowed, host=host
        )
        if int(game_result[0]) == int(game_result[1]) and not draw_allowed:
            winner = game[0] if game_result[0] > game_result[1] else game[1]
            fooracle.talk(
                f"{game[0]} vs. {game[1]} - {game_result[0]}:{game_result[1]} after 90 min, winner {winner}"
            )
        else:
            fooracle.talk(
                f"{game[0]} vs. {game[1]} - {game_result[0]}:{game_result[1]}"
            )

        results.append(game_result)

    return results


def foretell_league(
    fooracle: Fooracle,
    teams: list[str],
    return_match=False,
    host: str | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
    games: list[GameTeams] = []
    for ii in range(len(teams)):
        for jj in range(ii + 1, len(teams)):
            games.append((teams[ii], teams[jj]))

    if return_match:
        for ii in range(len(games)):
            games.append(games[ii][::-1])

    results = foretell_games(fooracle, games, host=host)
    return games, results


def foretell_playoff(
    fooracle: Fooracle,
    games: list[GameTeams],
    host: str | None = None,
) -> str:
    if len(games) > 1 and len(games) % 2 == 1:
        raise ValueError("Uneven number of games in tournament.")

    if len(games) == 1:
        print("Final")
    elif len(games) == 2:
        print("Semifinal")
    else:
        print("Round of best", len(games) * 2)

    results = foretell_games(fooracle, games, draw_allowed=False, host=host)
    if len(results) == 1:
        return get_winner(games[0], results[0])

    next_round: list[GameTeams] = []
    buffer = None
    for game, result in zip(games, results, strict=True):
        winner = get_winner(game, result)
        if buffer is None:
            buffer = winner
        else:
            next_round.append([buffer, winner])
            buffer = None

    print("")  # new line to separate for a nicer optic
    return foretell_playoff(fooracle, next_round, host)


def foretell_tournament(
    fooracle: Fooracle,
    groups: list[list[str]],
    host: str | None = None,
):
    # supports up to 32 groups ;-)
    group_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜΓΔΘ"
    standings = []
    finalists: list[GameTeams] = []

    if len(groups) > len(group_names):
        raise ValueError(f"Too many groups. Maximum is {len(group_names)}.")

    # Check if all teams are existing
    for group in groups:
        for team in group:
            if not fooracle.knows_team(team):
                raise ValueError(f"Unknown team: {team}")

    # calculate the group results as mini-leagues with no second leg game
    for group, name in zip(groups, group_names, strict=True):
        print("Group", name)
        games, results = foretell_league(fooracle, group, return_match=False, host=host)
        group_standing = calculate_league_standing(group, games, results)
        standings.append(group_standing)

        print("")

    for standing, name in zip(standings, group_names, strict=True):
        print("Group", name)
        print(standing)
        print("\n")

    # winner of group A [A] plays second of group B [b]
    # winner of group B [B] plays second of group A [a]
    # Arranged such that teams from the group phase meet at the finals again
    # AbCdEfGhBaDcFeHg for the standard 32 team world cup
    # winners = ACEG + BDFH
    # seconds = bdfh + aceg
    # So the zip pairs up the opposing teams
    winners = standings[::2] + standings[1::2]
    seconds = standings[1::2] + standings[::2]

    for winner, second in zip(winners, seconds, strict=True):
        finalists.append((str(winner.iloc[0].name), str(second.iloc[1].name)))

    champion = foretell_playoff(fooracle, finalists, host=host)

    print("Tournament champion:", champion)

    return champion


def main(
    data_file: Annotated[str, typer.Argument],
    team1: Annotated[str, typer.Argument] = "",
    team2: Annotated[str, typer.Argument] = "",
) -> None:
    data = pd.read_csv(data_file)
    foo = ConstantFooracle()
    foo.train(data)
    res = foo.foretell(team1, team2)
    print(res)


if __name__ == "__main__":
    typer.run(main)
