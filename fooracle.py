import sys
import warnings
import numpy as np
from scipy.stats import truncnorm, halfnorm, rv_continuous
import pandas as pd


def get_winner(game, result):
    if result[0] > result[1]:
        winner = game[0]
    elif result[0] < result[1]:
        winner = game[1]
    else:
        # if there is a draw we toss a coin
        winner = game[np.random.randint(2)]
    return winner


def calculate_league_standing(teams, games, results):
    standings = pd.DataFrame(
        index=teams, columns=["points", "goals_scored", "goals_taken"]
    ).fillna(0)
    for (game, result) in zip(games, results):
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


class fooracle:
    """Football oracle
    based on historical results it will tell you what the future brings.
    Uses data from: https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data
    """

    host = None
    criteria = None
    lower_bound = 0
    upper_bound = 30
    data = None
    parameter_home = None
    parameter_away = None
    minimal_sample_size = 5
    verbose = True
    quiet = True
    model: rv_continuous = None

    def __init__(self, data=None, use_criteria=True, host="Qatar"):
        self.talk("Welcome, you have summoned the fooracle!")

        if data is not None:
            self.load_data(data, use_criteria=use_criteria, host=host)

    def load_data(self, data, use_criteria, host):
        """hard coded for world cup 2018 in Russia
        Assumption: Only non-friendly games, games are on neutral territory or in Russia
        """
        neutral_territory = data.neutral is True
        in_host_country = (data.neutral is False) & (data.country == host)
        non_friendly = data.tournament != "Friendly"
        self.host = host

        self.criteria = (neutral_territory | in_host_country) & non_friendly
        if use_criteria:
            self.data = data[self.criteria]
        else:
            self.data = data
        self.talk(
            f"I can see the past, now I'm ready to tell the future for the tournament in {host}..."
        )

    def train_model(self, data=None):
        """Fits a truncated normal distribution to the home_score and away_score of the given data set (or the previously loaded data set).
        Returns the two parameter sets for home and away team"""
        self.parameter_home = self.fit_model(self.data[["home_score"]])
        self.parameter_away = self.fit_model(self.data[["away_score"]])
        return self.parameter_home, self.parameter_away

    def fit_model(self, scores):
        """Fits the truncated normal distribution to the score data.
        Returns the fitted parameters (a, b, loc, scale) for the scipy.truncnorm distribution"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.model == truncnorm:
                parameter = self.model.fit(scores, self.lower_bound, self.upper_bound)
            else:
                parameter = self.model.fit(scores)

        return parameter

    def train_model_on_teams(self, team1, team2):
        """Fitting the number of scored goals per game to the statistical model
        """
        team1_score = pd.concat(
            [
                self.data[self.data.home_team == team1]["home_score"],
                self.data[self.data.away_team == team1]["away_score"],
            ]
        )

        team2_score = pd.concat(
            [
                self.data[self.data.home_team == team2]["home_score"],
                self.data[self.data.away_team == team2]["away_score"],
            ]
        )

        if team1_score.size < self.minimal_sample_size:
            if not self.quiet:
                self.talk(
                    "For",
                    team1,
                    "the sample size is only",
                    team1_score.size,
                    ", fallback to overall home team score statistics",
                )
            team1_score = self.data["home_score"]

        if team2_score.size < self.minimal_sample_size:
            if not self.quiet:
                self.talk(
                    "For",
                    team2,
                    "the sample size is only",
                    team2_score.size,
                    ", fallback to overall away team score statistics",
                )
            team2_score = self.data["away_score"]

        team1_parameter = self.fit_model(team1_score)
        team2_parameter = self.fit_model(team2_score)
        return team1_parameter, team2_parameter

    def foretell(self, team1=None, team2=None, draw_allowed=True):
        if team1 is None or team2 is None:
            # self.talk('Your teams are incomprehensible. I will look into the future anyhow...')
            team1_parameter, team2_parameter = self.train_model()
        else:
            # self.talk('Good.', team1, 'vs.', team2, '- I will look into the future...')
            team1_parameter, team2_parameter = self.train_model_on_teams(team1, team2)

        team1_score = 0
        team2_score = 0

        # to make the while loop run exactly once if a draw is allowed we start with 9 which will be incremented to 10 and the loop condition is false
        if draw_allowed:
            counter = 9
        else:
            counter = 0

        while team1_score == team2_score and counter < 10:
            # b, loc, scale = team1_parameter
            team1_score = int(
                np.round(
                    self.model.rvs(
                        *team1_parameter[:-2],
                        loc=team1_parameter[-2],
                        scale=team1_parameter[-1],
                    )
                )
            )
            # b, loc, scale = team2_parameter
            team2_score = int(
                np.round(
                    self.model.rvs(
                        *team2_parameter[:-2],
                        loc=team2_parameter[-2],
                        scale=team2_parameter[-1],
                    )
                )
            )
            counter += 1

        self.talk(team1, "vs.", team2, "-", team1_score, ":", team2_score)
        return team1_score, team2_score

    def foretell_games(self, games, draw_allowed=True):
        """Predicts all results in the list games. Each entry must be a list of 2 countries."""
        results = []
        for game in games:
            game_result = self.foretell(game[0], game[1], draw_allowed=draw_allowed)
            results.append(game_result)

        return results

    def foretell_league(self, group, return_match=False):
        games = []
        for ii in range(len(group)):
            for jj in range(ii + 1, len(group)):
                games.append([group[ii], group[jj]])

        if return_match:
            for ii in range(len(games)):
                games.append(games[ii][::-1])

        results = self.foretell_games(games)
        return games, results

    def foretell_playoff(self, games):
        if len(games) > 1 and len(games) % 2 == 1:
            raise ValueError("Uneven number of games in tournament.")

        if len(games) == 1:
            print("Final")
        elif len(games) == 2:
            print("Semifinal")
        else:
            print("Round of best", len(games) * 2)

        results = self.foretell_games(games, draw_allowed=False)
        if len(results) == 1:
            return get_winner(games[0], results[0])

        next_round = []
        buffer = None
        for game, result in zip(games, results):
            winner = get_winner(game, result)
            if buffer is None:
                buffer = winner
            else:
                next_round.append([buffer, winner])
                buffer = None

        print("")  # new line to separate for a nicer optic
        return self.foretell_playoff(next_round)

    def foretell_tournament(self, groups, quiet=True, model=halfnorm):
        self.model = model

        # supports up to 32 groups ;-)
        group_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜΓΔΘ"
        standings = []
        finalists = []

        # Check if all teams are existing
        for g in groups:
            for t in g:
                if (
                    t not in self.data.home_team.values and t not in self.data.away_team.values
                ):
                    self.talk(
                        f"Warning! {t} not recognized as national team. Maybe a typo?"
                    )

        # stop error messages due to team occurences
        self.quiet = quiet

        # calculate the group results as mini-leagues with no second leg game
        for group, name in zip(groups, group_names):
            print("Group", name)
            games, results = self.foretell_league(group, return_match=False)
            group_standing = calculate_league_standing(group, games, results)
            standings.append(group_standing)
            print("")

        if self.verbose:
            for standing, name in zip(standings, group_names):
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

        for winner, second in zip(winners, seconds):
            finalists.append([winner.iloc[0].name, second.iloc[1].name])

        champion = self.foretell_playoff(finalists)

        print("Tournament champion:", champion)

        return champion

    def talk(self, *args):
        if self.verbose:
            print(*args)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        data_file = sys.argv[1]
        team1 = sys.argv[2]
        team2 = sys.argv[3]
    elif len(sys.argv) > 1:
        data_file = sys.argv[1]
        team1 = None
        team2 = None
    else:
        print("Usage: python fooracle.py datafile.csv [team1] [team2]")
        print(
            'For country names with blanks please use double quotes, e.g. "Korea Republic"'
        )

    data = pd.read_csv(data_file)
    foo = fooracle(data)
    res = foo.foretell(team1, team2)

    if not foo.verbose:
        print(res[0], res[1])
