import pandas as pd
import numpy as np
import plotly
import plotly.plotly as py
from plotly.plotly import iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import plotly.tools as tls
import cufflinks as cf

# # RMSE BY WINDOW LENGTH
# window_error = pd.read_csv('../images/rmse_by_window_len.csv')
#
# # Add data
# x_labels = window_error['window_len'].unique()
#
# LinReg = window_error.rmse[window_error.Model == 'LinReg']
# ElasticNet = window_error.rmse[window_error.Model == 'ElasticNet']
# BayesianRidge = window_error.rmse[window_error.Model == 'BayesianRidge']
# RandomForest = window_error.rmse[window_error.Model == 'RandomForest']
# GBR = window_error.rmse[window_error.Model == 'GBR']
#
# # Create and style traces
# trace0 = go.Scatter(
#     x = x_labels,
#     y = LinReg,
#     name = 'LinReg',
#     line = dict(
#         color = ('rgb(205, 12, 24)'),
#         width = 4)
# )
# trace1 = go.Scatter(
#     x = x_labels,
#     y = ElasticNet,
#     name = 'ElasticNet',
#     line = dict(
#         color = ('rgb(22, 96, 167)'),
#         width = 4,)
# )
# trace2 = go.Scatter(
#     x = x_labels,
#     y = BayesianRidge,
#     name = 'BayesianRidge',
#     line = dict(
#         color = ('rgb(0, 100, 80)'),
#         width = 4,)
# )
# trace3 = go.Scatter(
#     x = x_labels,
#     y = RandomForest,
#     name = 'RandomForest',
#     line = dict(
#         color = ('rgb(231, 107, 243)'),
#         width = 4,)
# )
# trace4 = go.Scatter(
#     x = x_labels,
#     y = GBR,
#     name = 'GBR',
#     line = dict(
#         color = ('rgb(0, 176, 246)'),
#         width = 4,)
# )
#
# data = [trace0, trace1, trace2, trace3, trace4]
#
# # Edit the layout
# layout = dict(title = 'RMSE by Window Length',
#               xaxis = dict(title = 'Roelling Avg Window Length'),
#               yaxis = dict(title = 'RMSE'),
#               )
#
# fig = dict(data=data, layout=layout)
# plotly.offline.plot(fig, filename='rmse_by_window_len.html')

# | Lineup             | Position | Predicted | Actual |  Diff |
# |--------------------|:--------:|:---------:|:------:|:-----:|
# | Jeff Teague        | PG       | 30.34     | 24.7   |  -5.6 |
# | Russell Westbrook  | PG       | 51.14     | 55.8   |  +4.7 |
# | Avery Bradley      | SG       | 19.6      | 24.4   |  +4.8 |
# | Garrett Temple     | SG       | 17.06     | 15.4   |  -1.7 |
# | Stanley Johnson    | SF       | 20.6      | 10.9   |  -9.7 |
# | LeBron James       | SF       | 51.71     | 54.9   |  +3.2 |
# | Bobby Portis       | PF       | 29.06     | 15.8   | -13.3 |
# | Frank Kaminsky     | PF       | 16.91*    | 4.7*   | -12.2 |
# | Karl-Anthony Towns | C        | 42.93     | 60.4   | +17.5 |
# | TOTAL              |          | 264.0     | 262.3  |  -1.7 |


fig = plt.figure()

df = pd.read_csv('../data/merged_df.csv')
target = df['player_fantasy_points']
target = target[target >3]
target = target.dropna()

mu = target.mean() # mean of distribution
sigma = target.std() # standard deviation of distribution

num_bins = 40
# the histogram of the data
n, bins, patches = plt.hist(target, num_bins, normed=1, facecolor='green', alpha=0.5)

# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Fantasy Points')
plt.ylabel('Probability')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)

plotly_fig = tls.mpl_to_plotly( fig )
plotly.offline.plot(plotly_fig, filename='histogram_over_three.html')
