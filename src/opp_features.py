import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _clean_columns(df):
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('#', '')
    to_drop = [x for x in df if x.endswith(('_y', 'pergame')) or x.startswith(('unnamed', 'date/time'))]
    df.drop(to_drop, axis=1, inplace=True)

    return df

def get_team_features(df):
    df = df.sort_values(by='team_game_date')
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

def get_opp_feature(df):
    df['opponent'] = np.where(
    df['team_team_name'] == df['team_home_team_name'],
    df['team_away_team_name'],
    df['team_home_team_name']
    )

    df = df.sort_values(by='team_game_date')

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

    return opp_df

def get_expanding_average(df, opp=True):
    if opp:
        df.groupby('opponent').expanding().mean()

    # at this point, need to add the date and other infor (maybe) back on
    # in order to merge with other dfs
    pass

def temp_plot():
# away_ppg =
# away_ppga =
away = np.array([124, 98, 128, 120, 149, 106, 86, 109, 92, 119, 102, 128, 129, 105, 130, 103, 115, 132, 127, 119])
away_mean = np.mean(away)
away_std = np.std(away)

# home_ppg =
# home_ppga =
home = np.array([119, 100, 126, 116, 103, 115, 110, 114, 99, 138, 112, 117, 105, 109, 95, 126, 102, 115, 123, 121, 116])
home_mean = np.mean(home)
home_std = np.std(home)
overs = []
combos = []
over = 228
for idx, away_score in enumerate(away):
    for i, home_score in enumerate(home):
        combos.append((idx, i))
        if away[idx] + home[i] > over:
            overs.append(away_score + home_score)
chance_over = len(overs) / len(combos)
chance_under = 1 - chance_over


def temp_merge(player_logs, dfs_logs):
    player_logs['player_id_game_id'] = (player_logs['player_game_id'].map(str) + player_logs['player_player_id'].map(str)).map(int)
    dfs_logs.dropna(subset=['player_#Game ID'], inplace=True)
    dfs_logs = _clean_columns(dfs_logs)
    # to_drop = [x for x in dfs_logs.columns if x in player_logs]
    dfs_logs['player_id_game_id'] = (dfs_logs['player_game_id'].map(int).map(str) + dfs_logs['player_player_id'].map(str)).map(int)
    # dfs_logs.drop(to_drop, axis=1, inplace=True)

    keys = ['player_id_game_id', 'team_game_id']
    merged_df = pd.merge(player_logs, dfs_logs, on=keys[0], how='inner', suffixes=('', '_y'))

    merged_df = merged_df.sort_values(by='player_game_date')
    merged_df['name'] = merged_df['player_firstname'] + ' ' + merged_df['player_lastname']
    merged_df = _clean_columns(merged_df)
    to_drop = [x for x in merged_df if x.startswith(('unnamed', 'Unnamed', 'player_date/time', 'team_date/time'))]
    merged_df = merged_df.drop(to_drop, axis=1)

    print(f"Data Merged\n---------------\nShape: {merged_df.shape}")
    print(f"\nDate Min: {merged_df.player_game_date.min()}")
    print(f"\nDate Max: {merged_df.player_game_date.max()}")

    return merged_df

if __name__ == '__main__':
    pd.set_option('display.max_columns', 200)
    # team_logs = pd.read_csv('../data/team_2019_gamelogs.csv', index_col=0)
    # team_logs = _clean_columns(team_logs)
    # team_df = get_team_features(team_logs)
    # opp_df = get_opp_feature(team_logs)
    # nuggets = opp_df[opp_df.opponent == 'Nuggets']

    # nugg_roll = get_rolling_averages(nuggets)

    '''
    PLAYER_LOG/DFS_LOG CLEANUP TO PERMANENTLY SAVE CLEANED RAW DATAFRAMES IN THE SAME
    FORMATS FOR MANIPULATION INTO TRAINING DF
    '''
    # dfslogs_2016 = pd.read_csv('../data/dfs_2016_logs.csv')
    # playerlogs_2016 = pd.read_csv('../data/player_2016_gamelogs.csv', index_col=0)
    # merged_2016 = temp_merge(playerlogs_2016, dfslogs_2016)
    #
    # dfslogs_2017 = pd.read_csv('../data/dfs_2017_logs.csv')
    # playerlogs_2017 = pd.read_csv('../data/player_2017_gamelogs.csv', index_col=0)
    # merged_2017 = temp_merge(playerlogs_2017, dfslogs_2017)
    #
    # dfslogs_2018 = pd.read_csv('../data/dfs_2018_logs.csv')
    # playerlogs_2018 = pd.read_csv('../data/player_2018_gamelogs.csv', index_col=0)
    # merged_2018 = temp_merge(playerlogs_2018, dfslogs_2018)

    dfslogs_2019 = pd.read_csv('../data/dfs_2019_logs.csv')
    playerlogs_2019 = pd.read_csv('../data/player_2019_gamelogs.csv', index_col=0)
    merged_2019 = temp_merge(playerlogs_2019, dfslogs_2019)

    # all_merged = [merged_2016, merged_2017, merged_2018, merged_2019]

    cols_to_drop = [x for x in merged_2019.columns if x not in merged_2016.columns]
    # cols_to_drop_2017 = [x for x in merged_2017.columns if x not in merged_2016.columns]
    # merged_2017.drop(cols_to_drop_2017, axis=1, inplace=True)
    # merged_2018.drop(cols_to_drop, axis=1, inplace=True)
    merged_2019.drop(cols_to_drop, axis=1, inplace=True)

    # Makes sure the dfs fit together
    merged_dfs = [merged_2016, merged_2017, merged_2018, merged_2019]
    merged = pd.concat(merged_dfs)

    # saves the DFs, not yet completed
    # merged_2016.to_csv('../data/playerlogs_2016.csv')
    # merged_2017.to_csv('../data/playerlogs_2017.csv')
    # merged_2018.to_csv('../data/playerlogs_2018.csv')
    merged_2019.to_csv('../data/playerlogs_2019.csv')

    '''
    TEAM_LOG CLEANUP TO PERMANENTLY SAVE CLEANED RAW DATAFRAMES IN THE SAME
    FORMAT FOR MANIPULATION INTO TRAINING DF
    '''
    # teamlogs_2016 = pd.read_csv('../data/team_2016_gamelogs.csv')
    # teamlogs_2016 = _clean_columns(teamlogs_2016)
    # opplogs_2016 = get_opp_feature(teamlogs_2016)
    #
    # teamlogs_2017 = pd.read_csv('../data/team_2017_gamelogs.csv')
    # teamlogs_2017 = _clean_columns(teamlogs_2017)
    # opplogs_2017 = get_opp_feature(teamlogs_2017)
    # to_drop_2017 = [x for x in teamlogs_2017.columns if x not in teamlogs_2016.columns]
    # teamlogs_2017.drop(to_drop_2017, axis=1, inplace=True)
    #
    # teamlogs_2018 = pd.read_csv('../data/team_2018_gamelogs.csv')
    # teamlogs_2018 = _clean_columns(teamlogs_2018)
    # opplogs_2018 = get_opp_feature(teamlogs_2018)
    # to_drop_2018 = [x for x in teamlogs_2018.columns if x not in teamlogs_2016.columns]
    # teamlogs_2018.drop(to_drop_2018, axis=1, inplace=True)

    teamlogs_2019 = pd.read_csv('../data/team_2019_gamelogs.csv')
    teamlogs_2019 = _clean_columns(teamlogs_2019)
    opplogs_2019 = get_opp_feature(teamlogs_2019)
    to_drop_2019 = [x for x in teamlogs_2019.columns if x not in teamlogs_2016.columns]
    teamlogs_2019.drop(to_drop_2019, axis=1, inplace=True)

    to_drop_2016 = [x for x in teamlogs_2016.columns if x not in teamlogs_2019.columns]
    teamlogs_2016.drop(to_drop_2016, axis=1, inplace=True)

    # Makes sure the dfs fit together
    # all_teamlogs = [teamlogs_2016, teamlogs_2017, teamlogs_2018, teamlogs_2019]
    # teamlogs = pd.concat(all_teamlogs)
    #
    # all_opplogs = [opplogs_2016, opplogs_2017, opplogs_2018, opplogs_2019]
    # opplogs = pd.concat(all_opplogs)

    # saves the DFs, not yet completed
    # teamlogs_2016.to_csv('../data/teamlogs_2016.csv')
    # teamlogs_2017.to_csv('../data/teamlogs_2017.csv')
    # teamlogs_2018.to_csv('../data/teamlogs_2018.csv')
    teamlogs_2019.to_csv('../data/teamlogs_2019.csv')

    '''
    NOW THAT TEAMLOGS AND PLAYERLOGS HAVE BEEN SAVED, TEAMLOGS CAN BE
    MANIPULATED TO ADD EXPANDING AVG STATISTICS FOR TEAM & OPP.

    OLD DATAFRAMES SHOULD BE REMOVED AND THE DATA LOADED NEEDS TO BE UPDATED
    TO SAVE THE 2019 DATAFRAME IN THE FORMATS ABOVE

    MAKE SURE OPPLOGS AND TEAMLOGS ARE IN DATE ORDER SO THEY CAN BE GROUPED
    AND AGGREGATED PROPERLY
    '''
