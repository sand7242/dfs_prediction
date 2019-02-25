import pandas as pd
import numpy as np
import datetime as dt
from tabulate import tabulate
from sklearn.ensemble import *
from sklearn.utils import shuffle
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import *
from DataLoader import DataLoader
from Featurizer import Featurizer
from scipy import stats

class Estimator:
    def __init__(self, df):
        self.df = df.copy(deep=True)
        self.info_cols = [
        'game_date',
        'name',
        'team',
        'opponent'
        ]
        self.train_df = self.df.drop(self.info_cols, axis=1)
        self.info_df = self.df.reindex(columns=self.info_cols)
        self.target = 'fantasy_points'
        self.index = 'player_id_game_id'

    def main(self):
        '''
        Training Data manipulation
        '''
        X_train, self.X_test, y_train, self.y_test = self.split_train_test(self.train_df, self.target)
        # self.test_outlier_removal(X_train, y_train, -1, 1, 60, 69)
        X_train_adj, y_train_adj = self.remove_training_outliers(X_train, y_train, 0, 79)
        X_train_pca = self.princ_comp_anal(X_train_adj, y_train_adj)

        '''
        Modeling w/ modified trainind data
        '''
        # self.xg_boost(X_train_pca, y_train_adj)
        # self.knn_regressor(X_train_adj, y_train_adj)
        # self.random_forest(X_train_adj, y_train_adj)
        self.bayesian_ridge(X_train_pca, y_train_adj)

    def split_train_test(self, train_df, target):
        y = train_df[target]
        X = train_df.drop(target, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, shuffle=False)
        self.total_train_obs = len(X_train)
        print(f"Number of original training observation: {self.total_train_obs}")

        return X_train, X_test, y_train, y_test

    def test_outlier_removal(self, X_train, y_train, min_lower=0, min_upper=0, max_lower=100, max_upper=100):
        '''
        Descr:  Function that allows to test a range of outliers to be removed from
                training dataframes. Outliers are not removed from test data.

        Input:  X_train, y_train from train_test_split
                min_lower: lower bound value for min values to be removed from train df
                min_upper:  upper bound value for min values to be removed from train df
                max_lower: lower bound value for max values to be removed from train df
                min_upper:  upper bound value for max values to be removed from train df
        Result: Passes min and max values to be removed from training df
                XGBoost model is then fit on modified training data
        '''
        for min in np.linspace(min_lower, min_upper, (min_upper - min_low + 1)):
            for max in np.linspace(max_lower, max_upper, (max_upper - max_lower + 1)):
                X_train_adj, y_train_adj = self.remove_training_outliers(X_train, y_train, min, max)
                self.xg_boost(X_train_adj, y_train_adj)

    def remove_training_outliers(self, X_train, y_train, min_target, max_target):
        '''
        Descr:  Function that removes outliers from training data.
                Outliers are NOT removed from test data.

        Input:  X_train, y_train from the train_test_split
        Output: X_train_adj, y_train_adj with outliers removed
        '''
        train_df = pd.merge(X_train, pd.DataFrame(y_train), on=self.index, how='left')
        train_df = train_df[train_df[self.target] > min_target]

        min_obs_removed =  self.total_train_obs - len(train_df)
        min_percent_removed = min_obs_removed / self.total_train_obs
        print(f"\n----------\nRemoving values less than {min_target}")
        print(f"* {min_obs_removed} low value observations removed or {round(min_percent_removed*100)}%")

        train_df = train_df[train_df[self.target] < max_target]

        max_obs_removed =  self.total_train_obs - len(train_df) - min_obs_removed
        max_percent_removed = max_obs_removed / self.total_train_obs
        print(f"Removing values greater than {max_target}")
        print(f"* {max_obs_removed} high value observations removed or {round(max_percent_removed*100)}%")

        y_train_adj = train_df[self.target]
        X_train_adj = train_df.drop(self.target, axis=1)

        return X_train_adj, y_train_adj

    def xg_boost(self, X_train, y_train):
        params = {
        "loss":["ls"],
        "learning_rate": [0.01],
        "min_samples_split": [0.06],
        "min_samples_leaf": [0.01],
        "max_depth":[2],
        "max_features":["auto"],
        "criterion": ["friedman_mse"],
        "subsample":[0.61],
        "n_estimators":[2000]
        }

        xgb = XGBRegressor()
        clf = GridSearchCV(xgb, params, cv=10, n_jobs=-1)

        clf.fit(X_train, y_train)
        print(clf.best_score_)
        # print(clf.best_params_)

        y_pred = clf.predict(self.X_test)
        y_pred = [round(value) for value in y_pred]
        rmse = round(mean_squared_error(self.y_test, y_pred) ** 0.5, 4)
        print(f"\n* XGBOOST RMSE: {rmse}\n")

    def random_forest(self, X_train, y_train):
        params = {
                 'bootstrap': [True],
                 'criterion': ['mse'],
                 'max_depth': [None],
                 'max_features': ['auto'],
                 'max_leaf_nodes': [None],
                 'min_impurity_decrease': [0.0],
                 'min_impurity_split': [None],
                 'min_samples_split': [2],
                 'n_estimators': [10,50,200],
                 'oob_score': [True],
                 'verbose': [0],
                 'warm_start': [False]
                 }

        rf = RandomForestRegressor()
        clf = GridSearchCV(rf, params, cv=10, n_jobs=-1)

        clf.fit(X_train, y_train)
        print(clf.best_score_)
        print(clf.best_params_)

        y_pred = clf.predict(self.X_test)
        y_pred = [round(value) for value in y_pred]
        rmse = round(mean_squared_error(self.y_test, y_pred) ** 0.5, 4)
        print(f"\n* Random Forest RMSE: {rmse}\n")
        self.plot_residuals(y_pred, rmse)
        self.plot_predicted_vs_actual(y_pred, rmse)

    def knn_regressor(self, X_train, y_train):
        params = {
        "n_neighbors":[12, 15, 18],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto"],
        "leaf_size": [30],
        }

        kn = KNeighborsRegressor()
        clf = GridSearchCV(kn, params, cv=10, n_jobs=-1)

        clf.fit(X_train, y_train)
        print(clf.best_score_)
        print(clf.best_params_)

        y_pred = clf.predict(self.X_test)
        y_pred = [round(value) for value in y_pred]
        rmse = round(mean_squared_error(self.y_test, y_pred) ** 0.5, 4)
        print(f"\n* KNeighbors Regressor RMSE: {rmse}\n")
        self.plot_residuals(y_pred, rmse)
        self.plot_predicted_vs_actual(y_pred, rmse)


    def bayesian_ridge(self, X_train, y_train):
        X_train = StandardScaler().fit_transform(X_train)
        clf = BayesianRidge(compute_score=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(self.X_test)
        y_pred = [round(value) for value in y_pred]
        rmse = round(mean_squared_error(self.y_test, y_pred) ** 0.5, 4)
        print(f"\n* Bayesian Ridge RMSE: {rmse}\n")

        ols = LinearRegression()
        ols.fit(X_train, y_train)
        y_pred = clf.predict(self.X_test)
        y_pred = [round(value) for value in y_pred]
        rmse = round(mean_squared_error(self.y_test, y_pred) ** 0.5, 4)
        print(f"\n* Linear Regression: {rmse}\n")

        self.bayesian_plots(clf, ols)

    def princ_comp_anal(self, X_train, y_train):
        X_train = StandardScaler().fit_transform(X_train)
        pca = PCA(n_components=33)
        principalComponents = pca.fit_transform(X_train)
        principalDf = pd.DataFrame(data=principalComponents)
        finalDf = pd.concat([principalDf, y_train], axis = 1)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.95) + 1
        print(d)

        plt.plot(cumsum)
        plt.show()

        return principalDf

    def plot_feature_importances(self, model):
        plot_importance(model)
        plt.show()

    def plot_residuals(self, y_pred, rmse):
        residuals = y_pred - self.y_test
        plt.figure(figsize=(10, 5))
        plt.scatter(self.y_test, residuals, s=20)
        plt.title(''.join(['Fantasy Points', ', Residual w/ Actual.', ' rmse = ', str(rmse)]))
        plt.xlabel('Actual Fantasy Points')
        plt.ylabel('Residuals')
        plt.tight_layout()
        plt.show()

    def plot_predicted_vs_actual(self, y_pred, rmse):
        plt.figure(figsize=(10, 5))
        plt.scatter(self.y_test, y_pred, s=20)
        plt.title(''.join(['Fantasy Points', ', Predicted vs. Actual', ' rmse = ', str(rmse)]))
        plt.xlabel('Actual Fantasy Points')
        plt.ylabel('Predicted Fantasy Points')
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)])
        plt.tight_layout()
        plt.show()

    def bayesian_plots(self, clf, ols):
        n_features = self.X_test.shape[1]
        relevant_features = np.random.randint(0, n_features, 10)
        w = np.zeros(n_features)
        lw = 2
        plt.figure(figsize=(6, 5))
        plt.title("Weights of the model")
        plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
                 label="Bayesian Ridge estimate")
        plt.plot(w, color='gold', linewidth=lw, label="Ground truth")
        plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")
        plt.xlabel("Features")
        plt.ylabel("Values of the weights")
        plt.legend(loc="best", prop=dict(size=12))

        plt.figure(figsize=(6, 5))
        plt.title("Histogram of the weights")
        plt.hist(clf.coef_, bins=n_features, color='gold', log=True,
                 edgecolor='black')
        plt.scatter(clf.coef_[relevant_features], np.full(len(relevant_features), 5.),
                    color='navy', label="Relevant features")
        plt.ylabel("Features")
        plt.xlabel("Values of the weights")
        plt.legend(loc="upper left")

        plt.figure(figsize=(6, 5))
        plt.title("Marginal log-likelihood")
        plt.plot(clf.scores_, color='navy', linewidth=lw)
        plt.ylabel("Score")
        plt.xlabel("Iterations")

    def create_residual_df(self):
        self.residual_df = self.X_test.merge(self.info_df, on='player_id_game_id', how='left')
        self.residual_df['actual_fps'] = self.y_test
        self.residual_df['predicted_fps'] = self.y_pred
        self.residual_df['residuals'] = round(self.residual_df['predicted_fps'] - self.residual_df['actual_fps'], 1)
        self.residual_df['percent_diff'] = round(abs((self.residual_df['residuals'] / self.residual_df['actual_fps']) * 100))
        self.residual_df = self.residual_df.sort_values(by=['percent_diff'], ascending=False)
        self.residual_df = pd.concat([self.residual_df, self.info_df], axis=1, sort=False)


if __name__ == '__main__':
    start_dt = dt.datetime.now()
    pd.set_option('display.max_columns', 200)

    train_df = pd.read_csv('../data/train_df.csv', index_col=0)
    estimator = Estimator(train_df)
    estimator.main()

    end_dt = dt.datetime.now()
    print(f"Time taken: {end_dt - start_dt}")
