import os
import io
import time
import base64
import requests
import pandas as pd
import datetime as dt
from bs4 import BeautifulSoup
from datetime import timedelta, date
from ohmysportsfeedspy import MySportsFeeds

class DataLoader():
    '''
    Class for pulling, merging and formatting multiple datasets
    '''
    def __init__(self):
        self.log_url = 'https://api.mysportsfeeds.com/v2.0/pull/nba/2018-2019-regular/date/DATE/CAT_gamelogs.csv?team=TEAM'
        self.dfs_url = 'https://api.mysportsfeeds.com/v2.0/pull/nba/2018-2019-regular/date/DATE/dfs.csv'
        self.teams = ['atl', 'bos', 'bro', 'cha', 'chi', 'cle', 'dal', 'den', 'det', \
                'gsw', 'hou', 'ind', 'lac', 'lal', 'mem', 'mia', 'mil', 'min', \
                'nop', 'nyk', 'okl', 'orl', 'phi', 'phx', 'por', 'sac', 'sas', \
                'tor', 'uta', 'was']

    def pull_data(self):
        self.player_logs = self.load_game_logs('player', self.log_url)
        self.team_logs = self.load_game_logs('team', self.log_url)
        self.load_daily_dfs_logs()
        self.merge_data(self.player_logs, self.dfs_logs, self.team_logs)


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

    def load_game_logs(self, category, url):
        df = pd.DataFrame()
        dates = self._daterange()

        for date in dates:
            for team in self.teams:
                response = self._send_request(url.replace('TEAM', team).replace('CAT', category).replace('DATE', date))
                team_df = pd.read_csv(io.StringIO(response.text), sep=',')
                print(f"{team.upper()} {category.upper()} logs loaded\n{len(team_df)} observations")
                df = pd.concat([df, team_df])
                time.sleep(5)

        to_drop = [x for x in df if x.startswith(('unnamed', 'date/time'))]
        df.drop(to_drop, axis=1, inplace=True)
        df.columns = [category + '_' + col for col in df.columns]

        path = '../data/temp_CAT_gamelogs.csv'
        df.to_csv(path.replace('CAT', category))

        print(f"\n{category.upper()} game logs loaded")
        print(f"\n----------\nFirst 5 Rows:\n {df.head()}")

        return df

    def load_daily_dfs_logs(self):
        self.dfs_logs = pd.DataFrame()
        dates = self._daterange()

        for date in dates:
            response = self._send_request(self.dfs_url.replace('DATE', date))
            single_day_df = pd.read_csv(io.StringIO(response.text), sep=',')
            print(f"{date} loaded\n{len(single_day_df)} observations")
            self.dfs_logs = pd.concat([self.dfs_logs, single_day_df]).dropna(axis=1, how='all')
            time.sleep(5)

        to_drop = [x for x in self.dfs_logs if x.startswith(('unnamed', 'date/time'))]
        self.dfs_logs.drop(to_drop, axis=1, inplace=True)
        self.dfs_logs.columns = ['player_' + col for col in self.dfs_logs.columns]
        self.dfs_logs.to_csv('../data/temp_dfs_logs.csv')

        print(f"\n----------\nDFS game logs loaded")
        print(f"\n----------\nFirst 5 Rows:\n {self.dfs_logs.head()}")

    def merge_data(self, player_logs, dfs_logs, team_logs):
        player_logs['player_game_id'] = (player_logs['player_#Game ID'].map(str) + player_logs['player_#Player ID'].map(str)).map(int)
        self.dfs_logs['player_game_id'] = (self.dfs_logs['player_#Game ID'].map(str) + self.dfs_logs['player_#Player ID'].map(str)).map(int)
        player_logs['team_game_id'] = (player_logs['player_#Game ID'].map(str) + player_logs['player_#Team ID'].map(str)).map(int)
        team_logs['team_game_id'] = (team_logs['team_#Game ID'].map(str) + team_logs['team_#Team ID'].map(str)).map(int)

        keys = ['player_game_id', 'team_game_id']
        self.merged_df = pd.merge(player_logs, self.dfs_logs, on=keys[0], how='inner', suffixes=('', '_y'))
        self.merged_df = pd.merge(self.merged_df, team_logs, on=keys[1], how='inner')

        self.merged_df = self._clean_columns(self.merged_df)
        self.merged_df = self.merged_df.sort_values(by='player_game_date')
        self.merged_df['name'] = self.merged_df['player_firstname'] + ' ' + self.merged_df['player_lastname']
        self.merged_df.to_csv('../data/temp_merged_df.csv')

        print(f"\n---------------\nFirst 5 Rows:\n {self.merged_df.head()}")
        print(f"\n---------------\nShape: {self.merged_df.shape}")

    def _clean_columns(self, df):
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('#', '')
        to_drop = [x for x in df if x.endswith(('_y', 'pergame')) or x.startswith(('unnamed', 'date/time'))]
        df.drop(to_drop, axis=1, inplace=True)

        return df

    def _daterange(self):

        today = dt.datetime.today()
        start_date = date(2018, 12, 12)
        end_date = date(today.year, today.month, today.day)
        dates  = []

        for n in range(int ((end_date - start_date).days)):
            dates.append((start_date + timedelta(n)).strftime('%Y%m%d'))

        return dates

if __name__ == '__main__':
    start_dt = dt.datetime.now()

    dl = DataLoader()
    dl.pull_data()

    end_dt = dt.datetime.now()
    print(f"\n---------------\nTime taken: {end_dt - start_dt}")
