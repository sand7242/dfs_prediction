import Knapsack
import pandas as pd
import datetime as dt
from Featurizer import Featurizer
from DataLoader import DataLoader


### NEED TO MAKE A DIRECTORY THAT HOLDS THE RAW DK AND FANDUEL DFs FROM EACH DAY ###
def format_for_prediction(late=None):
    today = dt.datetime.today().strftime('%Y%m%d')
    if late:
        late = '_late'
    else:
        late = ''
    # dk_path = '../data/daily_predictions/DK_' + today + late + '.csv'
    # fanduel_path = '../data/daily_predictions/fanduel_' + today + '.csv'
    dk_path = '../data/daily_predictions/DK_20181220.csv'
    fanduel_path = '../data/daily_predictions/fanduel_20181220.csv'
    dk = pd.read_csv(dk_path)
    fanduel = pd.read_csv(fanduel_path)

    train_df = pd.read_csv('../data/train_df.csv')
    keep_cols = [x for x in train_df.columns if not x.startswith('Unnamed:')]
    train_df = train_df[keep_cols]
    train_df = train_df.sort_values(by='date')
    train_df = train_df.dropna(subset=['name'])

    predict_df = pd.DataFrame()
    for player in train_df.name.unique():
         temp_df = train_df[train_df.name == player].iloc[[-1]]
         predict_df = pd.concat([predict_df, temp_df])

    fanduel['name'] = fanduel['Nickname'].map(str)
    fanduel = pd.merge(fanduel, predict_df, on='name', how='inner')
    fanduel['date'] = pd.to_datetime(fanduel['date'])
    fanduel = fanduel.sort_values(by='date')
    fanduel = fanduel[fanduel['Injury Indicator'] != 'GTD']
    fanduel = fanduel[fanduel['Injury Indicator'] != 'O']
    fanduel = fanduel[fanduel.date > '2018-12-01']

    # LEN(DK) BEFORE AND AFTER!!! TO CHECK THAT IT REMOVED INJURIES
    dk['name'] = dk['Name']
    predictions = pd.merge(fanduel, dk, on='name', how='inner', suffixes=['_fanduel', '_dk'])

    ###### HERE IS WHERE IT NEEDS TO BE ADJUSTED FOR FANDUEL ######
    for_predict = predictions.reindex(columns=['player_fg2ptmade', 'player_fg3ptmade', \
    'player_ftmade', 'player_reb', 'player_ast', 'player_tov', 'player_stl', \
    'player_blk', 'player_minseconds', 'player_salary', 'home']).values

    predictions['dk_predicted'] = dp.GBR.predict(for_predict)
    # predictions.to_csv('../data/daily_predictions/predictions' + today + '.csv')
    predictions.to_csv('../data/daily_predictions/predictions20181219.csv')
    for_knapsack = predictions.reindex(columns=['Position_dk', 'Name', 'Salary_dk', 'dk_predicted'])
    for_knapsack.to_csv('../data/daily_predictions/for_knapsack.csv')

    return for_knapsack


if __name__ == '__main__':
    start_dt = dt.datetime.now()

    # Saves raw dataset as merged_df.csv
    dl = DataLoader()
    dl.main()
    raw_path = '../data/merged_df.csv'
    for n in range(3, 4):
        # Saves training data as train_df.csv
        f = Featurizer(raw_path, window=n, dfs_provider='DraftKings')
        f.main()

        # Stores model created from training dataframe passed into it
        dp = DFSPredictor(f.df)
        dp.run_analysis()
        dp.to_markdown(dp.results_df)

    end_dt = dt.datetime.now()
    print(f"Time taken: {end_dt - start_dt}")


for_knapsack = for_knapsack[for_knapsack.Name != 'Jabari Parker']
for_knapsack.to_csv('../data/daily_predictions/for_knapsack.csv')
run Knapsack.py

dk = dk.reindex(columns=['player_fg2ptmade', 'player_fg3ptmade', \
'player_ftmade', 'player_reb', 'player_ast', 'player_tov', 'player_stl', \
'player_blk', 'player_minseconds', 'player_salary', 'home']).values

fanduel = fanduel.reindex(columns=['player_fg2ptmade', 'player_fg3ptmade', \
'player_ftmade', 'player_reb', 'player_ast', 'player_tov', 'player_stl', \
'player_blk', 'player_minseconds', 'player_salary', 'home']).values
