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

    def main(self):
        self.read_data()
        self.get_home_feature()
        self.format_training_data()
        self.get_rolling_averages()
        self.merge_final_df()

    def read_data(self):
        self.merged_2016 = pd.read_csv('../data/merged_2016_df.csv', index_col=0)
        self.merged_2017 = pd.read_csv('../data/merged_2017_df.csv', index_col=0)
        self.merged_2018 = pd.read_csv('../data/merged_2018_df.csv', index_col=0)
        self.merged_2019 = pd.read_csv('../data/merged_2019_df.csv', index_col=0)

        all_dfs = [self.merged_2016, self.merged_2017, self.merged_2018, self.merged_2019]
        self.df = pd.concat(all_dfs, sort=True)

        print(f"\n---------------\nFirst 5 Rows:\n {self.df.head()}")
        print(f"All data merged")
        print(f"Shape: {self.df.shape}")
        print(f"Date_Min: {self.df.player_game_date.min()}")
        print(f"Date_Min: {self.df.player_game_date.max()}")

        self.df = self.df.sort_values(by='player_game_date')
        self.df = self.df[self.df.player_dfs_type == self.dfs_provider]
        print(f"\n----------\nstripping to FANDUEL")
        print(f"DF SHAPE: {self.df.shape}\n----------\n")

    def _rolling_average(self, df, window):
        return df.rolling(window=window).mean().shift(1)

    def get_home_feature(self):
        self.df['home'] = 1 * (self.df['player_home_team_abbr.'] == self.df['player_team_abbr.'])
        print(f"\n----------\ngetting HOME FEATURE")
        print(f"DF SHAPE: {self.df.shape}\n----------\n")

    def format_training_data(self):
        self.df = self.df[self.df.player_fantasy_points > 3]
        print(f"\n----------\nremoving obs. with less than 3 points")
        print(f"DF SHAPE: {self.df.shape}\n----------\n")
        self.stats_df = self.df.reindex(columns=[
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
        print(f"\n----------\ncreating stats_df")
        print(f"stats_df columns:\n{self.stats_df.columns}")
        print(f"STATS_DF SHAPE: {self.stats_df.shape}\n----------\n")


        self.info_df = self.df.reindex(columns=[
        'player_player_id',
        'player_game_date',
        'name',
        'home',
        'player_position',
        'player_fantasy_points'
        ])
        print(f"\n----------\ncreating info_df")
        print(f"info_df columns:\n{self.info_df.columns}")
        print(f"info_DF SHAPE: {self.info_df.shape}\n----------\n")

    def get_rolling_averages(self):
        self.stats_df = self.stats_df.reset_index(drop=True)
        print(f"\n----------\nresetting index for stats_df")
        print(self.stats_df.head())
        print(f"STATS_DF SHAPE: {self.stats_df.shape}\n----------\n")
        self.stats_df = self.stats_df.groupby('player_player_id').apply(lambda x: self._rolling_average(x, self.window))
        print(f"\n----------\ngetting rolling averages for stats_df")
        print(self.stats_df.head())
        print(f"STATS_DF SHAPE: {self.stats_df.shape}\n----------\n")

    def merge_final_df(self):
        self.info_df = self.info_df.reset_index(drop=True)
        print(f"\n----------\nresetting index for INFO_df")
        print(self.info_df.head())
        print(f"INFO_DF SHAPE: {self.info_df.shape}\n----------\n")
        self.merged = pd.concat([self.stats_df, self.info_df], axis=1, sort=False)
        print(f"\n----------\ncreating merged df")
        print(self.merged.head())
        print(f"MERGED_DF SHAPE: {self.merged.shape}\n----------\n")
        self.merged = self.merged.iloc[:, 1:]
        print(f"\n----------\ndropping janky first column")
        print(self.merged.head())
        print(f"MERGED_DF SHAPE: {self.merged.shape}\n----------\n")
        self.merged.rename(index=str, columns={
                                    "player_position": "position",
                                    "player_fantasy_points": "fantasy_points",
                                    "player_game_date": "date",
                                    "player_player_id": "player_id"
                                    }, inplace=True)
        print(f"\n----------\nrenaming cols in merged_df")
        print(self.merged.head())
        print(f"MERGED_DF SHAPE: {self.merged.shape}\n----------\n")
        self.merged = self.merged.dropna(axis=0, how='any').reset_index(drop=True)
        print(f"\n----------\ndropping rows with Nans")
        print(self.merged.head())
        print(f"MERGED_DF SHAPE: {self.merged.shape}\n----------\n")
        self.merged = self.merged.dropna(subset=['player_salary'])
        self.merged.to_csv('../data/train_df.csv')
        print(f"\n----------\nsaving training data from merged_df")
        print(f"MERGED_DF SHAPE: {self.merged.shape}\n----------\n")

if __name__ == '__main__':
    start_dt = dt.datetime.now()

    # saves the raw dataframe
    dl = DataLoader()
    dl.main()

    # saves the training dataset
    featurizer = Featurizer(dfs_provider='FanDuel')
    featurizer.main()

    print(f"\n----------\nLOADING TRAINING DATA")
    temp = pd.read_csv('../data/train_df.csv', index_col=0)
    print(temp.head())

    end_dt = dt.datetime.now()
    print(f"Time taken: {end_dt - start_dt}")
