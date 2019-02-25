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
    def __init__(self, dfs_provider):
        self.dfs_provider = dfs_provider
        self.years = ['2016', '2017', '2018', '2019']
        self.player_path = '../data/playerlogs_YEAR.csv'
        self.team_path = '../data/teamlogs_YEAR.csv'
        self.playerlogs = {}
        self.teamlogs = {}
        self.opplogs = {}
        self.final_playerlogs = pd.DataFrame()
        self.final_opplogs = pd.DataFrame()
        self.final_teamlogs = pd.DataFrame()

    def main(self):
        # load playerlogs
        # get home feature on playerlogs
        # format player features
        # combine playerlogs into 1 playerlogs df
        self.load_playerlogs()

        # load teamlogs
        # format opp_features from teamlogs
        # format teamlogs
        self.load_teamlogs()

        # merge playerlogs, teamlogs, opplogs
        self.merge_final_df()
        self.format_final_df()

    def load_playerlogs(self):

        for year in self.years:
            self.playerlogs[year] = pd.read_csv(self.player_path.replace('YEAR', year), index_col=0)
            self.playerlogs[year] = self.playerlogs[year][self.playerlogs[year]['player_dfs_type'] == self.dfs_provider]
            self.playerlogs[year] = self.get_home_feature(self.playerlogs[year])
            self.playerlogs[year] = self.get_df_dummies(self.playerlogs[year], cols=['player_position'])
            self.playerlogs[year] = self.format_player_features(self.playerlogs[year])
            self.final_playerlogs = pd.concat([self.final_playerlogs, self.playerlogs[year]], sort=False)

    def load_teamlogs(self):

        for year in self.years:
            self.teamlogs[year] = pd.read_csv(self.team_path.replace('YEAR', year), index_col=0)
            self.opplogs[year] = self.format_opp_features(self.teamlogs[year])
            self.final_opplogs = pd.concat([self.final_opplogs, self.opplogs[year]], sort=False)

            self.teamlogs[year] = self.format_team_features(self.teamlogs[year])
            self.final_teamlogs = pd.concat([self.final_teamlogs, self.teamlogs[year]], sort=False)

    def get_home_feature(self, df):
        df['home'] = np.where(
        df['player_home_team_abbr.'] == df['player_team_abbr.'], 1, 0
        )

        return df

    def get_df_dummies(self, df, cols):
        '''
        gettin dummies
        '''
        df = pd.get_dummies(df, columns=cols, drop_first=True)

        return df

    def format_player_features(self, df):
        df = df.sort_values(by='player_game_date')

        player_stats = [
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
        ]

        player_info = [
        'player_player_id',
        'player_team_id',
        'player_game_id',
        'player_game_date',
        'name',
        'home',
        'player_position_F',
        'player_position_G',
        'player_position_PF',
        'player_position_PG',
        'player_position_SF',
        'player_position_SG',
        'player_fantasy_points',
        'team_id_game_id',
        'player_id_game_id'
        ]

        player_stats_df = df.reindex(columns=player_stats)
        player_stats_df = self.get_rolling_averages(player_stats_df, groupby='player_player_id', window=3)

        player_info_df = df.reindex(columns=player_info)
        player_info_df = player_info_df.reset_index(drop=True)

        player_df = pd.concat([player_stats_df, player_info_df], axis=1, sort=False)

        return player_df

    def format_opp_features(self, df):

        df = df.sort_values(by='team_game_id')
        df = df.drop('team_id_game_id', axis=1)
        df.columns = df.columns.str.replace('team', 'opp')

        opp_stats = [
        'opponent_id',
        'opp_fg2ptmade',
        'opp_fg3ptmade',
        'opp_ftmade',
        'opp_reb',
        'opp_ast',
        'opp_tov',
        'opp_stl',
        'opp_blk'
        ]

        opp_info = [
        'opponent',
        'opp_game_date',
        'opp_game_id',
        'opp_id_game_id'
        ]

        opp_stats_df = df.reindex(columns=opp_stats)
        opp_stats_df = self.get_rolling_averages(opp_stats_df, groupby='opponent_id', window=3)

        opp_info_df = df.reindex(columns=opp_info)
        opp_info_df = opp_info_df.reset_index(drop=True)

        opp_df = pd.concat([opp_stats_df, opp_info_df], axis=1, sort=False)

        return opp_df

    def format_team_features(self, df):
        df = df.sort_values(by='team_game_id')

        team_stats = [
        'team_team_id',
        'team_fg2ptmade',
        'team_fg3ptmade',
        'team_ftmade',
        'team_reb',
        'team_ast',
        'team_tov',
        'team_stl',
        'team_blk'
        ]

        team_info = [
        'team_team_abbr.',
        'team_game_date',
        'team_game_id',
        'team_id_game_id',
        'opp_id_game_id'
        ]

        team_stats_df = df.reindex(columns=team_stats)
        team_stats_df = self.get_rolling_averages(team_stats_df, groupby='team_team_id', window=3)

        team_info_df = df.reindex(columns=team_info)
        team_info_df = team_info_df.reset_index(drop=True)

        team_df = pd.concat([team_stats_df, team_info_df], axis=1, sort=False)

        return team_df

    def merge_final_df(self):
        team_key = "team_id_game_id"
        self.result = self.final_playerlogs.merge(self.final_teamlogs, on=team_key, how='left')

        opp_key = "opp_id_game_id"
        self.result = self.result.merge(self.final_opplogs, on=opp_key, how='left')

    def format_final_df(self):

        self.result = self.result.iloc[:, 1:]
        self.result.rename(index=str, columns={
                                    "player_player_id": "player_id",
                                    "player_fantasy_points": "fantasy_points",
                                    "player_game_date": "game_date",
                                    "team_team_abbr.": "team"
                                    }, inplace=True)

        final_numeric = [
        'player_id_game_id',
        'player_fg2ptmade',
        'player_fg3ptmade',
        'player_ftmade',
        'player_reb',
        'player_ast',
        'player_tov',
        'player_stl',
        'player_blk',
        'player_minseconds',
        'player_salary',
        'player_position_F',
        'player_position_G',
        'player_position_PF',
        'player_position_PG',
        'player_position_SF',
        'player_position_SG',
        'team_fg2ptmade',
        'team_fg3ptmade',
        'team_ftmade',
        'team_reb',
        'team_ast',
        'team_tov',
        'team_stl',
        'team_blk',
        'home',
        'opp_fg2ptmade',
        'opp_fg3ptmade',
        'opp_ftmade',
        'opp_reb',
        'opp_ast',
        'opp_tov',
        'opp_stl',
        'opp_blk',
        'fantasy_points'
        ]

        final_info = [
        'game_date',
        'name',
        'team',
        'opponent'
        ]

        self.numeric_df = self.result.reindex(columns=final_numeric)
        self.info_df = self.result.reindex(columns=final_info)

        self.result = pd.concat([self.numeric_df, self.info_df], axis=1, sort=False)
        self.result = self.result.dropna(axis=0, how='any').set_index('player_id_game_id')

        self.result.head()
        self.result.to_csv('../data/train_df.csv')

    def get_rolling_averages(self, df, groupby, window):
        player_df = df.reset_index(drop=True)
        player_df = player_df.groupby(groupby).apply(lambda x: self._rolling_average(x, window))

        return player_df

    def _rolling_average(self, df, window):
        return df.rolling(window=window).mean().shift(1)

if __name__ == '__main__':
    start_dt = dt.datetime.now()
    pd.set_option('display.max_columns', 200)

    # pulls and saves the raw dataframes
    # dl = DataLoader()
    # dl.main()

    # saves the training dataset
    featurizer = Featurizer(dfs_provider='FanDuel')
    featurizer.main()

    # featurizer.result[(featurizer.result['name'] == 'Ryan Anderson') & (featurizer.result['team_game_date'] == '2016-02-23')]

    # print(f"\n----------\nLOADING TRAINING DATA")
    # temp = pd.read_csv('../data/train_df.csv', index_col=0)
    # print(temp.head())
    # print(temp.shape)

    # nuggets_opp = featurizer.opp_logs['2019'][featurizer.opp_logs['2019'].opponent == 'Nuggets']
    # nuggets_opp[nuggets_opp.opp_game_date == '2018-12-31']
    #
    # nuggets_team = featurizer.teamlogs['2019'][featurizer.teamlogs['2019'].team_team_name == 'Nuggets']
    # nuggets_team[nuggets_team.team_game_date == '2018-12-31']

    # print(nuggets_opp)
    # print(nuggets_team)


    end_dt = dt.datetime.now()
    print(f"Time taken: {end_dt - start_dt}")
