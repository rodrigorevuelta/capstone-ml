from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

def predict(payload):
    """Scales Payload"""
    LOG.info("Scaling Payload: \n %s",payload)
    #scaler = StandardScaler().fit(payload.astype(float))
    #scaled_adhoc_predict = scaler.transform(payload.astype(float))
    return 1.0

@app.route("/")
def home():
    html = f"<h3>Stock Market Prediction Home</h3>"
    return html.format(format)

@app.route("/predict", methods=['POST'])
def predict():
    json_payload = request.json
    LOG.info("JSON payload: \n%s",json_payload)
    inference_payload = pd.DataFrame(json_payload)
    LOG.info("Inference payload DataFrame: \n%s",inference_payload)
    # scale the input
    prediction = predict(inference_payload)
    # get an output prediction from the pretrained model, clf
    #prediction = list(clf.predict(scaled_payload))
    # Log the output prediction value
    LOG.info('Prediction value: %s',prediction)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    # load pretrained model as clf
    #clf = joblib.load("./model_data/boston_housing_prediction.joblib")
    app.run(host='0.0.0.0', port=80, debug=False) # specify port=80
