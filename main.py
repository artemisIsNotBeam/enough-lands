import numpy as np
from tensorflow import keras
import pandas as pd
#from tensorflow.keras import layers

import tensorflow as tf
#import tensorflow_probability as tfp
from flask import Flask, render_template,request,redirect


import sklearn
# let's pull our handy linear fitter from our 'prediction' toolbox: sklearn!
from sklearn.linear_model import LinearRegression

train = pd.read_csv("data/lands.csv")
test = pd.read_csv("data/testlands.csv")

def get_features_labels(df):
  #gets the 1st and 2nd column
  features=df.values[:,[0,1]]
  labels = df.values[:,2]

  return features, labels

trainFeatures, trainLabels= get_features_labels(train)
testFeatures, testLabels= get_features_labels(test)

trainFeatures = trainFeatures.astype('int')
trainLabels = trainLabels.astype('int32')

# set up our model
linear = LinearRegression()

# train the model 
linear.fit(trainFeatures,trainLabels)

# test the model
X_new = np.array([[60, 65]])  # create a new data point
y_pred = linear.predict(X_new)  # predict the label for the new data point

def useModel(cards,mv):
  X_new = np.array([[cards, mv]])
  value = linear.predict(X_new)[0]
  cheap = round(linear.predict(X_new)[0])
  return cheap, value 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/guide')
def guide():
    return render_template("guide.html")

@app.route("/predict", methods= ['POST', 'GET'])
def predict():
    output = request.form.to_dict()
    if output["lands"] == '' or output['mv']=='':
        return render_template("goBack.html")

    lands = float(output["lands"])
    mv = float(output["mv"])

    should,exact = useModel(lands,mv)
    # should go like {should} {exact}
    return render_template("predict.html", should=should,exact = exact)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False,port=5000)