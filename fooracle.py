import sys
import warnings
import numpy as np
from scipy.stats import truncnorm
import pandas as pd

def get_winner( game, result ):
    if result[0] > result[1]:
        winner = game[0]
    elif result[0] < result[1]:
        winner = game[1]
    else:
        #TODO: use some more sophisticated method to solve the draw than this
        winner = game[0]
    return winner

def calculate_league_standing( teams, games, results ):
    standings = pd.DataFrame(index = teams, columns=['points', 'goals_scored', 'goals_taken']).fillna(0)
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
    standings['goal_difference'] = standings['goals_scored'] - standings['goals_taken']
    standings = standings.sort_values(by=['points', 'goal_difference', 'goals_scored'], ascending=False)
    return standings


class fooracle:
    """Football oracle
    based on historical results it will tell you what the future brings.
    Uses data from: https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017/data
    """
    guests = None
    host = None
    criteria = None
    lower_bound = 0
    upper_bound = 30
    data = None
    parameter_home = None
    parameter_away = None
    minimal_sample_size = 5
    verbose = True

    def __init__( self, data = None ):
        self.talk('Welcome, you have summoned the fooracle!')
        if not data is None:
            self.load_data( data )

    def load_data(self, data):
        """ hard coded for world cup 2018 in Russia
        Assumption: Only non-friendly games, games are on neutral territory or in Russia
        """
        neutral_territory = (data.neutral == True)
        in_host_country = ((data.neutral == False) & (data.country == 'Russia'))
        non_friendly = (data.tournament != 'Friendly')
        self.guests = ['Iceland', 'Sweden', 'Denmark', 'England', 'Belgium', 'France', 'Germany', 'Switzerland', 'Croatia', 'Serbia', 'Spain', 'Portugal', 'Poland', 'Mexico', 'Costa Rica', 'Panama', 'Colombia', 'Peru', 'Brazil', 'Uruguay', 'Argentina', 'Senegal', 'Nigeria', 'Morocco', 'Tunisia', 'Egypt', 'Saudi Arabia', 'Iran', 'Korea Republic', 'Japan', 'Australia']
        self.host = 'Russia'
        participants = ( data.home_team.isin( self.guests + [self.host] ) & ( data.away_team.isin( self.guests + [self.host] ) ) )
        self.criteria = ((neutral_territory | in_host_country) & non_friendly) & participants
        self.data = data[self.criteria]
        self.talk('I can see the past, now I\'m ready to tell the future...')

    def train_model( self, data = None ):
        """Fits a truncated normal distribution to the home_score and away_score of the given data set (or the previously loaded data set).
        Returns the two parameter sets for home and away team"""
        self.parameter_home = self.fit_model(self.data[['home_score']])
        self.parameter_away = self.fit_model(self.data[['away_score']])
        return self.parameter_home, self.parameter_away

    def fit_model( self, scores ):
        """Fits the truncated normal distribution to the score data.
        Returns the fitted parameters (a, b, loc, scale) for the scipy.truncnorm distribution"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parameter = truncnorm.fit(scores, self.lower_bound, self.upper_bound)
        return parameter

    def train_model_on_teams( self, team1, team2 ):
        """
        """
        team1_score = self.data[self.data.home_team == team1]['home_score'].append(self.data[self.data.away_team == team1]['away_score'])
        team2_score = self.data[self.data.home_team == team2]['home_score'].append(self.data[self.data.away_team == team2]['away_score'])

        if team1_score.size < self.minimal_sample_size:
            self.talk('For', team1, 'the sample size is only', team1_score.size, ', fallback to overall home team score statistics')
            team1_score = self.data['home_score']

        if team2_score.size < self.minimal_sample_size:
            self.talk('For', team2, 'the sample size is only', team2_score.size, ', fallback to overall away team score statistics')
            team2_score = self.data['away_score']   
                 
        team1_parameter = self.fit_model( team1_score )
        team2_parameter = self.fit_model( team2_score )
        return team1_parameter, team2_parameter


    def foretell( self, team1 = None, team2 = None ):
        if team1 is None or team2 is None:
            #self.talk('Your teams are incomprehensible. I will look into the future anyhow...')
            team1_parameter, team2_parameter = self.train_model()
        else:
            #self.talk('Good.', team1, 'vs.', team2, '- I will look into the future...')
            team1_parameter, team2_parameter = self.train_model_on_teams( team1, team2 )

        a, b, loc, scale = team1_parameter
        team1_score = int(np.round(truncnorm.rvs(a, b, loc=loc, scale=scale)))
        a, b, loc, scale = team2_parameter
        team2_score = int(np.round(truncnorm.rvs(a, b, loc=loc, scale=scale)))
        self.talk( team1, 'vs.', team2, '-',  team1_score, ':', team2_score)
        return team1_score, team2_score

    def foretell_games( self, games ):
        """Predicts all results in the list games. Each entry must be a list of 2 countries.
        """
        results = []
        for game in games:
            game_result = self.foretell( game[0], game[1] )
            results.append(game_result)

        return results

    def foretell_league( self, group, return_match = False ):
        games = []
        for ii in range(len(group)):
            for jj in range(ii+1, len(group)):
                games.append( [ group[ii], group[jj] ] )

        if return_match:
            for ii in range(len(games)):
                games.append(games[ii][::-1])
        
        results = self.foretell_games( games )
        return games, results

    def foretell_playoff( self, games ):
        if len(games) > 1 and len(games)%2 == 1:
            raise ValueError('Uneven number of games in tournament.')
        
        results = self.foretell_games( games )
        if len(results) == 1:
            return get_winner( games[0], results[0] )

        next_round = []
        buffer = None
        for game, result in zip(games, results):
            winner = get_winner(game, result)
            if buffer is None:
                buffer = winner
            else:
                next_round.append( [ buffer, winner ] )
                buffer = None
        
        return self.foretell_playoff( next_round )

    def foretell_tournament( self, groups ):
        group_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        standings = []
        finalists = []
        for group in groups:
            games, results = self.foretell_league( group )
            group_standing = calculate_league_standing( group, games, results )
            standings.append(group_standing)
        
        if self.verbose:
            for standing, name in zip(standings, group_names):
                print('Group', name)
                print(standing)
                print('\n')
    
        winners = standings[::2] + standings[1::2]
        seconds = standings[1::2] + standings[::2]
        
        for winner, second in zip(winners, seconds):
            finalists.append([winner.iloc[0].name, second.iloc[1].name])

        

        champion = self.foretell_playoff( finalists )

        print('Winner world cup 2018:', champion)

        return champion

    def talk( self, *args ):
        if self.verbose:
            print(*args)
     

if __name__ == '__main__':
    if len(sys.argv) > 3:
        data_file = sys.argv[1]
        team1 = sys.argv[2]
        team2 = sys.argv[3]
    elif len(sys.argv) > 1:
        data_file = sys.argv[1]
        team1 = None
        team2 = None
    else:
        print('Usage: python fooracle.py datafile.csv [team1] [team2]')
        print('For country names with blanks please use double quotes, e.g. "Korea Republic"')

    data = pd.read_csv(data_file)
    foo = fooracle(data)
    res = foo.foretell( team1, team2 )

    if not foo.verbose:
        print(res[0], res[1])
