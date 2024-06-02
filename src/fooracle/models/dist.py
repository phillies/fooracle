from dataclasses import dataclass

import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from scipy.stats import halfnorm  # type: ignore[import-untyped]

from fooracle.data import HistoricDataSchema
from fooracle.logger import LOGGER
from fooracle.models.base import Fooracle


def get_winner_from_row(row: Series) -> str:
    """
    Determines the winner of a match based on the scores in a row of a pandas DataFrame.

    Args:
        row (pd.Series): A row of a pandas DataFrame containing the scores of a match.

    Returns:
        str | None: The name of the winning team, or None if it's a draw.
    """
    if row.home_score > row.away_score:
        return row.home_team
    elif row.away_score > row.home_score:
        return row.away_team
    else:
        return ""


@dataclass
class DistFooracle(Fooracle):
    teams_data: DataFrame

    def _get_parameter(self, team, side):
        if np.isnan(self.teams.loc[team, f"{side}_loc"]) or np.isnan(
            self.teams.loc[team, f"{side}_scale"]
        ):
            if np.isnan(self.teams.loc[team, "total_loc"]) or np.isnan(
                self.teams.loc[team, "total_scale"]
            ):
                return [0, 1]
            else:
                return [
                    self.teams.loc[team, "total_loc"],
                    self.teams.loc[team, "total_scale"],
                ]
        else:
            return [
                self.teams.loc[team, f"{side}_loc"],
                self.teams.loc[team, f"{side}_scale"],
            ]

    def foretell(
        self, team1: str, team2: str, draw_allowed: bool = True, host: str | None = None
    ) -> tuple[int, int]:
        if team1 == host or host is None:
            team1_parameter = self._get_parameter(team1, "home")
        else:
            team1_parameter = self._get_parameter(team1, "away")
        if team2 == host:
            team2_parameter = self._get_parameter(team2, "home")
        else:
            team2_parameter = self._get_parameter(team2, "away")

        team1_score = 0
        team2_score = 0

        # to make the while loop run exactly once if a draw is allowed we start with 9 which will be incremented to 10 and the loop condition is false
        if draw_allowed:
            counter = 9
        else:
            counter = 0

        while team1_score == team2_score and counter < 10:
            # b, loc, scale = team1_parameter
            # We use halfnorm, which does not have a b
            team1_score = int(
                np.round(
                    halfnorm.rvs(
                        # *team1_parameter[:-2],
                        loc=team1_parameter[-2],
                        scale=team1_parameter[-1],
                    )
                )
            )
            # b, loc, scale = team2_parameter
            team2_score = int(
                np.round(
                    halfnorm.rvs(
                        # *team2_parameter[:-2],
                        loc=team2_parameter[-2],
                        scale=team2_parameter[-1],
                    )
                )
            )
            counter += 1

        return team1_score, team2_score

    def train(self, game_data: DataFrame) -> None:
        HistoricDataSchema.validate(game_data)
        data = game_data.dropna()

        data.loc[:, "winner"] = data.apply(get_winner_from_row, axis=1)

        # create a series with the country names as index and the number of games played
        countries = pd.concat([data.home_team, data.away_team]).value_counts()
        # create a series with the country names as index and the number of games won
        winners = data.winner.value_counts()

        if "" in winners.index:
            winners.drop("", inplace=True)

        teams = pd.merge(
            countries.rename("games"),
            winners.rename("wins"),
            left_index=True,
            right_index=True,
            how="left",
        ).fillna(0)
        teams.loc[:, "win_pct"] = teams.apply(lambda x: x.wins / x.games, axis=1)

        for team, _row in teams.index:
            home_score = data[data.home_team == team].home_score.dropna()
            away_score = data[data.away_team == team].away_score.dropna()
            total_score = pd.concat([home_score, away_score])
            if len(home_score) > 10:
                loc, scale = halfnorm.fit(home_score.values)
                teams.loc[team, "home_loc"] = loc
                teams.loc[team, "home_scale"] = scale
            if len(away_score) > 10:
                loc, scale = halfnorm.fit(away_score.values)
                teams.loc[team, "away_loc"] = loc
                teams.loc[team, "away_scale"] = scale
            if len(total_score) > 10:
                loc, scale = halfnorm.fit(total_score.values)
                teams.loc[team, "total_loc"] = loc
                teams.loc[team, "total_scale"] = scale
        self.teams_data = teams

    def store(self, path: str) -> None:
        try:
            self.teams_data.to_csv(path)
        except Exception as e:
            LOGGER.error(f"Unable to write to {path}: {e}")

    def load(self, path: str) -> None:
        try:
            self.teams_data = pd.read_csv(path, index_col=0)
        except Exception as e:
            LOGGER.error(f"Unable to load from {path}: {e}")

    def knows_team(self, team: str) -> bool:
        return team in self.teams_data.index
