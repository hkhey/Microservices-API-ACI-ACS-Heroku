# score.pyimport json
import os
import joblib
import logging
import json

import pandas as pd
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)
def run(data):
    logging.info("model 2: request received")
    test_data = pd.DataFrame(json.loads(json.loads(data)['input']))
    
    proba = model.predict_proba(test_data)[:,-1].tolist()
    logging.info("Request processed")
    return json.dumps({'proba':proba})
