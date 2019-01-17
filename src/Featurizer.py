import os
import time
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from numpy.random import seed
from DataLoader import DataLoader

class Featurizer():
    '''
    class for creating predictive feaures from raw data
    '''
    def __init__(self, dfs_provider, window=3):
        self.window = window
        self.dfs_provider = dfs_provider
        self.years = ['2016', '2017', '2018', '2019']
        self.player_path = '../data/playerlogs_YEAR.csv'
        self.team_path = '../data/teamlogs_YEAR.csv'

    def main(self):
        self.read_stat_logs()
        # self.merge_final_df()

    def read_stat_logs(self):
        '''
        This needs to be its own class for reading in each data set. Each dataset
        will then need to be sent through the featurizer.
        '''
        self.player_logs = {}
        self.team_logs = {}
        self.opp_logs = {}

        # NEW IDEA for year in years, create temp team/player dfs and append those dfs
        # to the self.playerlogs/self.teamlogs to make it easier to merge final dfs
        # This current idea doesn't work for team and opp dfs. opp dfs need to be indpenednent
        for year in self.years:
            self.player_logs[year] = pd.read_csv(self.player_path.replace('YEAR', year), index_col=0)
            self.player_logs[year] = self.player_logs[year][self.player_logs[year].player_dfs_type == self.dfs_provider]
            self.player_logs[year] = self.get_home_feature(self.player_logs[year])
            self.player_logs[year] = self.format_player_data(self.player_logs[year])

        for year in self.years:
            self.team_logs[year] = pd.read_csv(self.team_path.replace('YEAR', year), index_col=0)
            self.opp_logs[year] = self.get_opp_features(self.team_logs[year])
            self.team_logs[year] = self.get_team_features(self.team_logs[year])

        # self.playerlogs = pd.concat(player_logs, sort=True)
        # self.playerlogs = playerlogs[playerlogs.player_dfs_type == dfs_provider]

        # self.teamlogs = pd.concat(player_logs, sort=True)

    def _rolling_average(self, df, window):
        return df.rolling(window=window).mean().shift(1)

    def get_home_feature(self, df):
        df['home'] = 1 * (df['player_home_team_abbr.'] == df['player_team_abbr.'])
        return df

    def format_player_data(self, df):
        df = df.sort_values(by='player_game_date')
        df = df[df.player_fantasy_points > 3]

        player_df = df.reindex(columns=[
        'player_player_id',
        'player_fg2ptmade',
        'player_fg3ptmade',
        'player_ftmade',
        'player_reb',
        'player_ast',
        'player_tov',
        'player_stl',
        'player_blk',
        'player_minseconds',
        'player_salary'
        ])
        player_df = self.get_rolling_averages(player_df)

        info_df = df.reindex(columns=[
        'player_player_id',
        'player_game_date',
        'name',
        'home',
        'player_position',
        'player_fantasy_points'
        ])
        info_df = info_df.reset_index(drop=True)

        player_df = pd.concat([player_df, info_df], axis=1)

        return player_df

    def get_rolling_averages(self, df):
        player_df = df.reset_index(drop=True)
        player_df = player_df.groupby('player_player_id').apply(lambda x: self._rolling_average(x, self.window))

        return player_df

    def get_team_features(self, df):
        df = df.sort_values(by='team_game_id')

        team_df = df.reindex(columns=[
        'team_team_name',
        'team_fg2ptmade',
        'team_fg3ptmade',
        'team_ftmade',
        'team_reb',
        'team_ast',
        'team_tov',
        'team_stl',
        'team_blk'
        ])

        info_df = df.reindex(columns=[
        'team_game_date',
        'team_game_id'
        ])

        team_df = team_df.groupby('team_team_name').expanding().mean()
        team_df = team_df.reset_index(level=0).reset_index(drop=True)
        team_df = pd.concat([team_df, info_df], axis=1)

        return team_df

    def get_opp_features(self, df):
        df['opponent'] = np.where(
        df['team_team_name'] == df['team_home_team_name'],
        df['team_away_team_name'],
        df['team_home_team_name']
        )

        df = df.sort_values(by='team_game_id')

        df.columns = df.columns.str.replace('team', 'opp')
        opp_df = df.reindex(columns=[
        'opponent',
        'opp_fg2ptmade',
        'opp_fg3ptmade',
        'opp_ftmade',
        'opp_reb',
        'opp_ast',
        'opp_tov',
        'opp_stl',
        'opp_blk'
        ])

        info_df = df.reindex(columns=[
        'opp_game_date',
        'opp_game_id'
        ])

        opp_df = opp_df.groupby('opponent').expanding().mean()
        opp_df = opp_df.reset_index(level=0).reset_index(drop=True)
        opp_df = pd.concat([opp_df, info_df], axis=1)


        return opp_df

    def merge_final_df(self):
        self.merged = self.merged.iloc[:, 1:]
        self.merged.rename(index=str, columns={
                                    "player_position": "position",
                                    "player_fantasy_points": "fantasy_points",
                                    "player_game_date": "date",
                                    "player_player_id": "player_id"
                                    }, inplace=True)
        self.merged = self.merged.dropna(axis=0, how='any').reset_index(drop=True)
        self.merged = self.merged.dropna(subset=['player_salary'])
        self.merged.to_csv('../data/train_df.csv')

if __name__ == '__main__':
    start_dt = dt.datetime.now()
    pd.set_option('display.max_columns', 200)

    # pulls and saves the raw dataframes
    # dl = DataLoader()
    # dl.main()

    # saves the training dataset
    featurizer = Featurizer(dfs_provider='FanDuel')
    featurizer.main()

    # print(f"\n----------\nLOADING TRAINING DATA")
    # temp = pd.read_csv('../data/train_df.csv', index_col=0)
    # print(temp.head())
    # print(temp.shape)

    nuggets_opp = featurizer.opp_logs['2019'][featurizer.opp_logs['2019'].opponent == 'Nuggets']
    nuggets_opp[nuggets_opp.opp_game_date == '2018-12-31']

    nuggets_team = featurizer.team_logs['2019'][featurizer.team_logs['2019'].team_team_name == 'Nuggets']
    nuggets_team[nuggets_team.team_game_date == '2018-12-31']

    print(nuggets_opp)
    print(nuggets_team)


    end_dt = dt.datetime.now()
    print(f"Time taken: {end_dt - start_dt}")
