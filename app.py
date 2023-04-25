import pandas as pd
import numpy as np
from tensorflow import keras
#from tensorflow.keras import layers

import tensorflow as tf
#import tensorflow_probability as tfp

import pandas as pd
import numpy as np
from flask import Flask, render_template,request,redirect

#from tensorflow import keras

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

import sklearn
# let's pull our handy linear fitter from our 'prediction' toolbox: sklearn!
from sklearn.linear_model import LinearRegression

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
    port = 5000
    print(f"The app is running on port {port}")
    app.run(port=port, debug=True)