import os
import io
import time
import base64
import requests
import numpy as np
import pandas as pd
import datetime as dt
from bs4 import BeautifulSoup
from datetime import timedelta, date
from ohmysportsfeedspy import MySportsFeeds

# TEAMLOGS TODO
# Teamlogs are loaded on their own and drop cols that startwith(team_foul, team_ejections)
# Change so

class DataLoader():
    '''
    Class for pulling, merging and formatting multiple datasets
    '''
    def __init__(self):
        self.log_url = 'https://api.mysportsfeeds.com/v2.0/pull/nba/2018-2019-regular/CAT_gamelogs.csv?team=TEAM'
        self.dfs_url = 'https://api.mysportsfeeds.com/v2.0/pull/nba/2018-2019-regular/date/DATE/dfs.csv'
        self.teams = ['atl', 'bos', 'bro', 'cha', 'chi', 'cle', 'dal', 'den', 'det', \
                'gsw', 'hou', 'ind', 'lac', 'lal', 'mem', 'mia', 'mil', 'min', \
                'nop', 'nyk', 'okl', 'orl', 'phi', 'phx', 'por', 'sac', 'sas', \
                'tor', 'uta', 'was']
        self.teamlogs = pd.DataFrame()
        self.playerlogs = pd.DataFrame()
        self.dfs_logs = pd.DataFrame()
        self.dates = self._daterange()

    def main(self):
        self.load_teamlogs()
        self.format_teamlogs()
        self.get_opp_feature()

        self.load_dfs_logs()
        self.load_playerlogs()
        self.format_playerlogs()

        self.merge_playerlogs()
        self.format_final_playerlogs()

    def _send_request(self, url):
        try:
            response = requests.get(
                url=url,
                params={
                    "fordate": ''.join(str(dt.datetime.now())[:10].split('-'))
                },
                headers={
                    "Authorization": "Basic " + base64.b64encode('{}:{}'.format(os.environ['MSF_API_KEY'],'MYSPORTSFEEDS').encode('utf-8')).decode('ascii')
                }
            )
            print(f'\nResponse HTTP Status Code: {response.status_code}')
        except requests.exceptions.RequestException:
            print('HTTP Request failed')

        return response

    def load_teamlogs(self):
        for team in self.teams:
            response = self._send_request(self.log_url.replace('TEAM', team).replace('CAT', 'team'))
            team_df = pd.read_csv(io.StringIO(response.text), sep=',')
            print(f"{team.upper()} teamlogs loaded\n{len(team_df)} observations")
            self.teamlogs = pd.concat([self.teamlogs, team_df], sort=True)
            time.sleep(5)

        print(f"\n----------\n teamlogs loaded\nShape: {self.teamlogs.shape}")

    def format_teamlogs(self):
        self.teamlogs.columns = ['team_' + col for col in self.teamlogs.columns]
        self.teamlogs = self._clean_columns(self.teamlogs)

        to_drop = [x for x in self.teamlogs if x.startswith((
        'team_unnamed',
        'team_date/time',
        'team_foulf',
        'team_foulsdrawn',
        'team_foulpersdrawn',
        'team_foultechdrawn',
        'team_ejections'
        ))]

        self.teamlogs.drop(to_drop, axis=1, inplace=True)

        self.teamlogs['team_id_game_id'] = (self.teamlogs['team_team_id'].map(str) + self.playerlogs['team_game_id'].map(str)).map(int)

    def get_opp_feature(self):
        self.teamlogs['opponent'] = np.where(
        self.teamlogs['team_team_name'] == self.teamlogs['team_home_team_name'],
        self.teamlogs['team_away_team_abbr.'],
        self.teamlogs['team_home_team_abbr.']
        )

        self.teamlogs['opponent_id'] = np.where(
        self.teamlogs['team_team_name'] == self.teamlogs['team_home_team_name'],
        self.teamlogs['team_away_team_id'],
        self.teamlogs['team_home_team_id']
        )

        self.teamlogs.to_csv('../data/teamlogs_2019.csv')

    def load_dfs_logs(self):
        for date in self.dates:
            response = self._send_request(self.dfs_url.replace('DATE', date))
            single_day_df = pd.read_csv(io.StringIO(response.text), sep=',')
            print(f"{date} loaded\n{len(single_day_df)} observations")

            self.dfs_logs = pd.concat([self.dfs_logs, single_day_df], sort=True).dropna(axis=1, how='all')
            time.sleep(5)

        print(f"\n----------\nDFS game logs loaded\nShape: {self.dfs_logs.shape}")

    def load_playerlogs(self):
        for team in self.teams:
            response = self._send_request(self.log_url.replace('TEAM', team).replace('CAT', 'player'))
            team_df = pd.read_csv(io.StringIO(response.text), sep=',')
            print(f"{team.upper()} playerlogs loaded\n{len(team_df)} observations")
            self.playerlogs = pd.concat([self.playerlogs, team_df], sort=True)
            time.sleep(5)

        print(f"\n----------\n playerlogs loaded\nShape: {self.playerlogs.shape}")

    def format_playerlogs(self):
        self.dfs_logs = self._clean_columns(self.dfs_logs)
        self.dfs_logs.columns = ['player_' + col for col in self.dfs_logs.columns]

        self.playerlogs = self._clean_columns(self.playerlogs)
        self.playerlogs.columns = ['player_' + col for col in self.playerlogs.columns]

    def merge_playerlogs(self):
        self.playerlogs['player_id_game_id'] = (self.playerlogs['player_player_id'].map(str) + self.playerlogs['player_game_id'].map(str)).map(int)
        self.dfs_logs['player_id_game_id'] = (self.dfs_logs['player_player_id'].map(str) + self.dfs_logs['player_game_id'].map(str)).map(int)

        key = 'player_id_game_id'
        self.playerlogs = pd.merge(self.playerlogs, self.dfs_logs, on=key, how='inner', suffixes=('', '_y'))
        self.playerlogs = self.playerlogs.sort_values(by='player_game_date')
        self.playerlogs = self.playerlogs.reset_index(drop=True)

    def format_final_playerlogs(self):
        to_drop = [x for x in self.playerlogs if x.endswith(('_y', 'pergame')) or x.startswith((
        'player_date/time',
        'player_gamesstarted',
        'player_fouls',
        'player_foulpersdrawn',
        'player_foult',
        'player_foulf',
        'player_ejections'
        ))]

        self.playerlogs.drop(to_drop, axis=1, inplace=True)

        self.playerlogs['name'] = self.playerlogs['player_firstname'] + ' ' + self.playerlogs['player_lastname']
        self.playerlogs['team_id_game_id'] = (self.playerlogs['player_team_id'].map(str) + self.playerlogs['player_game_id'].map(str)).map(int)

        self.playerlogs.to_csv('../data/playerlogs_2019.csv')

    def _clean_columns(self, df):
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('#', '')

        return df

    def _daterange(self):
        today = dt.datetime.today()
        start_date = date(2018, 10, 17)
        end_date = date(today.year, today.month, today.day)
        dates  = []

        for n in range(int ((end_date - start_date).days)):
            dates.append((start_date + timedelta(n)).strftime('%Y%m%d'))

        return dates


if __name__ == '__main__':
    pd.set_option('display.max_columns', 200)
    start_dt = dt.datetime.now()

    dl = DataLoader()
    dl.main()

    end_dt = dt.datetime.now()
    print(f"\n---------------\nTime taken: {end_dt - start_dt}")
