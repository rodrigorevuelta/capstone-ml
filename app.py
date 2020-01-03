from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import math
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series,DataFrame
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

@app.route("/")
def home():
  html = "<h3>Stock Market Prediction Home</h3>"
  return html.format(format)

def predictStock(inputData):
  start = datetime.datetime(inputData["initYear"],1,inputData["initMonth"])
  end = datetime.datetime(inputData["endYear"],1,inputData["endMonth"])
  df = web.DataReader('AAPL','yahoo',start,end)
  close_px = df['Adj Close']
  mavg = close_px.rolling(window=100).mean()
  rets = close_px / close_px.shift(1) - 1
  dfreg = df.loc[:,['Adj Close','Volume']]
  dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
  dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
  # Drop missing value
  dfreg.fillna(value=-99999, inplace=True)
  # We want to separate 1 percent of the data to forecast
  forecast_out = int(math.ceil(0.01 * len(dfreg)))
  # Separating the label here, we want to predict the AdjClose
  forecast_col = 'Adj Close'
  dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
  X = np.array(dfreg.drop(['label'], 1))
  # Scale the X so that everyone can have the same distribution for linear regression
  X = preprocessing.scale(X)
  # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
  X_lately = X[-forecast_out:]
  X = X[:-forecast_out]
  # Separate label and identify it as y
  y = np.array(dfreg['label'])
  y = y[:-forecast_out]
  # Linear regression
  clfreg = LinearRegression(n_jobs=-1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  clfreg.fit(X_train, y_train)
  # Quadratic Regression 2
  clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
  clfpoly2.fit(X_train, y_train)
  # Quadratic Regression 3
  clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
  clfpoly3.fit(X_train, y_train)
  # KNN Regression
  clfknn = KNeighborsRegressor(n_neighbors=2)
  clfknn.fit(X_train, y_train)
  confidencereg = clfreg.score(X_test, y_test)
  confidencepoly2 = clfpoly2.score(X_test,y_test)
  confidencepoly3 = clfpoly3.score(X_test,y_test)
  confidenceknn = clfknn.score(X_test, y_test)
  #clf = svm.SVC(gamma=0.001, C=100.)
  #clf.fit(X_train, y_train)
  #forecast_set = clf.predict(X_lately)
  #dfreg['Forecast'] = np.nan
  prediction = clfreg.predict(X_train)
  value = prediction[0]
  return value



@app.route("/predict", methods=['GET'])
def predict():
  # Log the output prediction value
  inputData = {}
  inputData["initYear"] = request.args.get('initYear',type=int)
  inputData["initMonth"] = request.args.get('initMonth',type=int)
  inputData["endYear"] = request.args.get('endYear',type=int)
  inputData["endMonth"] = request.args.get('endMonth',type=int)
  LOG.info(inputData)
  prediction = predictStock(inputData)
  LOG.info('Prediction value: %s',prediction)
  return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False) # specify port=80