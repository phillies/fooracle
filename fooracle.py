import sys
import numpy as np
from scipy.stats import halfnorm
import pandas as pd
import os
import torch
from torch import nn


def get_winner_from_row(row):
    if row.home_score > row.away_score:
        return row.home_team
    elif row.away_score > row.home_score:
        return row.away_team
    else:
        return None


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
        result = np.round(result)
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


class MLP(nn.Module):
    '''
    Multilayer Perceptron for simple regression.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(633, 1024),
            nn.RReLU(),
            nn.Linear(1024, 128),
            nn.RReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, game_noise=None):
        '''
            Forward pass
        '''
        x = self.layers(x)

        return x


class fooracle:
    """Football oracle
    based on historical results it will tell you what the future brings.
    Uses data from: https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data
    """

    host = None
    data = None
    parameter_home = None
    parameter_away = None
    minimal_sample_size = 5
    verbose = True
    quiet = True
    mlp = None
    fairy_dust = 0.0
    teams_encoder = dict()

    def __init__(self, data: pd.DataFrame | str = "teams.fth", host="Qatar", fairy_dust=0.0, mlp_data='state_dict.pth', mode='mlp'):
        """Creates the fooracle

        Args:
            data (pd.DataFrame | str, optional): Kaggle results.csv data to train the distributions or feather file with pre-fit data. Defaults to teams.fth.
            host (str, optional): _description_. Defaults to "Qatar".
            fairy_dust (float, optional): Sprinkle fairy dust on the results. Adds random normal noise to the results. 0.0 is 0 noise, 1.0 is 100% noise. Defaults to 0.0.
            mlp_data (str, optional): _description_. Defaults to 'state_mlp3.pth'.
            mode (str, optional): if multilayer perceptron (mlp) or distribution fitting (dist) should be used. Defaults to mlp.
        """
        self.talk("Welcome, you have summoned the fooracle!")
        self.host = host
        self.fairy_dust = fairy_dust
        self.mode = mode

        if type(data) is pd.DataFrame():
            self.load_games(data)
        else:
            self.load_data(data)

        if os.path.exists(mlp_data):
            self.mlp = MLP()
            self.mlp.load_state_dict(torch.load(mlp_data))
            self.mlp.eval()

        self.talk(
            f"I can see the past, now I'm ready to tell the future for the tournament in {host}..."
        )

    def load_data(self, filename):
        if os.path.exists(filename):
            self.teams = (
                pd.read_feather(filename).set_index("index").rename_axis(None, axis=0)
            )
            self.teams_encoder = dict()
            for t, row in self.teams.iterrows():
                self.teams_encoder[t] = np.array([int(x) for x in row.encoded.split(';')])

    def load_games(self, data):
        """hard coded for world cup 2018 in Russia
        Assumption: Only non-friendly games, games are on neutral territory or in Russia
        """
        self.data = data

        self.data["winner"] = self.data.apply(get_winner_from_row, axis=1)

        countries = pd.concat([self.data.home_team, self.data.away_team]).value_counts()
        winners = self.data.winner.value_counts()

        self.teams = pd.merge(
            countries.rename("games"),
            winners.rename("wins"),
            left_index=True,
            right_index=True,
            how="left",
        ).fillna(0)
        self.teams["win_pct"] = self.teamsteams.apply(lambda x: x.wins / x.games, axis=1)

        for t, row in self.teams.iterrows():
            home_score = self.data[self.data.home_team == t].home_score.dropna()
            away_score = self.data[self.data.away_team == t].away_score.dropna()
            total_score = pd.concat([home_score, away_score])
            if len(home_score) > 10:
                loc, scale = halfnorm.fit(home_score.values)
                self.teamsteams.loc[t, "home_loc"] = loc
                self.teamsteams.loc[t, "home_scale"] = scale
            if len(away_score) > 10:
                loc, scale = halfnorm.fit(away_score.values)
                self.teamsteams.loc[t, "away_loc"] = loc
                self.teamsteams.loc[t, "away_scale"] = scale
            if len(total_score) > 10:
                loc, scale = halfnorm.fit(total_score.values)
                self.teamsteams.loc[t, "total_loc"] = loc
                self.teamsteams.loc[t, "total_scale"] = scale

    def get_parameter(self, team, side):
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

    def foretell(self, team1=None, team2=None, draw_allowed=True):
        if self.mode == 'mlp':
            return self.foretell_mlp(team1, team2, draw_allowed)
        else:
            return self.foretell_dist(team1, team2, draw_allowed)

    def foretell_mlp(self, team1=None, team2=None, draw_allowed=True):
        if team1 == self.host:
            neutral = 0
        elif team2 == self.host:
            neutral = 0
            team2 = team1
            team1 = self.host
        else:
            neutral = 1
        game = np.hstack([
            self.teams_encoder[team1].reshape(1, -1),
            self.teams_encoder[team2].reshape(1, -1),
            np.array([neutral]).reshape(1, 1)
        ])
        res = self.mlp(torch.Tensor(game)).detach().numpy().flatten()
        res = np.maximum((1 - self.fairy_dust) * res + self.fairy_dust * np.random.randn(2) * 3.0, 0)
        return res[0], res[1]

    def foretell_dist(self, team1=None, team2=None, draw_allowed=True):
        if team1 is None or team2 is None:
            # self.talk('Your teams are incomprehensible. I will look into the future anyhow...')
            return
        else:
            # self.talk('Good.', team1, 'vs.', team2, '- I will look into the future...')
            if team1 == self.host or self.host is None:
                team1_parameter = self.get_parameter(team1, "home")
            else:
                team1_parameter = self.get_parameter(team1, "away")
            if team2 == self.host:
                team2_parameter = self.get_parameter(team2, "home")
            else:
                team2_parameter = self.get_parameter(team2, "away")

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

    def foretell_games(self, games, draw_allowed=True):
        """Predicts all results in the list games. Each entry must be a list of 2 countries."""
        results = []
        for game in games:
            game_result = self.foretell(game[0], game[1], draw_allowed=draw_allowed)
            if (int(game_result[0].round()) == int(game_result[1].round()) and not draw_allowed):
                winner = game[0] if game_result[0] > game_result[1] else game[1]
                self.talk(game[0], "vs.", game[1], "-", int(game_result[0].round()), ":", int(game_result[1].round()), "after 90 min, winner ", winner)
            else:
                self.talk(game[0], "vs.", game[1], "-", int(game_result[0].round()), ":", int(game_result[1].round()))

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

    def foretell_tournament(self, groups, quiet=True):
        # supports up to 32 groups ;-)
        group_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜΓΔΘ"
        standings = []
        finalists = []

        # Check if all teams are existing
        for g in groups:
            for t in g:
                if t not in self.teams.index:
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
