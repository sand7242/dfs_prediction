import numpy as np
import pandas as pd
import datetime as dt
from numpy.random import seed
from tabulate import tabulate
from sklearn.ensemble import *
from Featurizer import Featurizer
from DataLoader import DataLoader
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet

class DFSPredictor():
    '''
    class for handling model building and new data predictions
    '''
    def __init__(self, df):
        self.df = df
        self.results_df = pd.DataFrame()
        self.models = [LinearRegression, ElasticNet, BayesianRidge, RandomForestRegressor, GradientBoostingRegressor]
        self.model_names = ['LinReg', 'ElasticNet', 'BayesianRidge', 'RandomForest', 'GBR']
        self.metrics = [mean_squared_error]
        self.metric_names = ['RMSE']

    def run_analysis(self):
        self.split_train_test()
        # dp.run_models()

    # def read_data(self, path):
    #     self.df = pd.read_csv(self.df_path)
    #     keep_cols = [x for x in self.df.columns if not x.startswith('Unnamed:')]
    #     self.df = self.df[keep_cols]

    def split_train_test(self):
        print(f"\n----------\nsplitting data into train/test")
        print(f"ORIGINAL DF: {self.df.head()}")
        print(f"ORIGINAL DF SHAPE: {self.df.shape}\n----------\n")
        self.y = self.df.pop('fantasy_points').values
        print(f"\n----------\nTARGET DATA: {self.y}")
        print(self.y.shape)
        self.X = self.df.drop(['name', 'date', 'player_id', 'position'], axis=1).values
        print(f"\n----------\nTRAINING DATA: {self.X}")
        print(self.X.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                self.X, self.y, random_state=1)

    def run_models(self):
        for model, m_name in zip(self.models, self.model_names):
            setattr(self, m_name, model)
            results = pd.Series()
            results['Model'] = m_name
            self.model = getattr(self, m_name)
            self.model = self.make_model(self.model)
            for score, s_name in zip(self.metrics, self.metric_names):
                y_pred = self.model.predict(self.X_test)
                results['rmse'] = mean_squared_error(self.y_test, y_pred) ** 0.5
                results['window_len'] = f.window
                print(results)
            setattr(self, m_name, self.model)
            self.results_df = self.results_df.append(results, ignore_index=True)

    def run_grid_search(self):
        self.parameters = {
        "loss":["ls"],
        "learning_rate": [0.001, 0.005, 0.01],
        "min_samples_split": [1, 2],
        "min_samples_leaf": [2, 3, 4],
        "max_depth":[2, 3],
        "max_features":["auto"],
        "criterion": ["friedman_mse"],
        "subsample":[0.7, 0.8, 0.9],
        "n_estimators":[1000]
        }

        results = pd.Series()

        self.clf = GridSearchCV(GradientBoostingRegressor(), self.parameters, cv=10, n_jobs=-1)
        self.clf.fit(self.X_train, self.y_train)
        y_pred = self.clf.predict(self.X_test)
        results['Model'] = 'GS GBR'
        results['rmse'] = mean_squared_error(self.y_test, y_pred) ** 0.5
        results['window_len'] = f.window
        print(results)
        print(self.clf.score(self.X_train, self.y_train))
        print(self.clf.best_params_)
        self.results_df = self.results_df.append(results, ignore_index=True)

    def format_for_prediction(df):
        y = df.pop('target')


    def make_model(self, classifier, **kwargs):
        '''
        Make specified sklearn model
        args:
        classifier (object): sklearn model object
        X_train (2d numpy array): X_train matrix from train test split
        y_train (1d numpy array): y_train matrix from train test split
        **kwargs (keyword arguments): key word arguments for specific sklearn model
        '''
        model = classifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        return model

    @staticmethod
    def to_markdown(df, round_places=3):
        """Returns a markdown, rounded representation of a dataframe"""
        print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe', showindex=False))

if __name__ == '__main__':
    start_dt = dt.datetime.now()
    raw_path = '../data/merged_df.csv'
<<<<<<< HEAD

    # Saves training data as train_df.csv
    fd = Featurizer(raw_path, window=3, dfs_provider='FanDuel')
    fd.main()

    # Stores model created from training dataframe passed into it
    dpfd = DFSPredictor(fd.merged)
    dpfd.run_analysis()
    dpfd.to_markdown(dpfd.results_df)
=======
    df_path = '../data/train_df.csv'

    # featurizers = []
    # windows = range(1, 10)
    #
    # for window in windows:
    #     featurizer = Featurizer(raw_path, window)
    #     featurizer.main()
    #     featurizers.append(featurizer)
    #
    # predictors = []
    # window_error = pd.DataFrame()

    # for f in featurizers:
    #     dp = DFSPredictor(df_path)
    #     print(f"\n----------\n{f.df.head()}")
    #     dp.run_analysis()
    #     print(f"\n----------\nWINDOW SIZE: {f.window}")
    #     dp.to_markdown(dp.results_df)
    #     window_error = pd.concat([window_error, dp.results_df])

    # saves train_df.csv
    f = Featurizer(raw_path, 3)
    f.main()

    dp = DFSPredictor(df_path)
    dp.run_analysis()
    print('Running Models')
    dp.run_models()
    print('Running Grid Search')
    dp.run_grid_search()
    dp.to_markdown(dp.results_df)
>>>>>>> 2a6cac5440f9b2f3cf3cc4bdea47de65b74a5672

    end_dt = dt.datetime.now()
    print(f"Time taken: {end_dt - start_dt}")
