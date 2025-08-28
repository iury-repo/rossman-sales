import pandas as pd
from xgboost import XGBRegressor
from flask import Flask, Response, request
from preprocessing.Rossman import RossmanPreprocessing

# Loading model
model = XGBRegressor()
model.load_model('model/xgb_rossman.json') 

# Initializing API
app = Flask(__name__)

@app.route('/rossman/predict', methods = ['POST'])
def rossman_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):                     # If there's only one example
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns= test_json[0].keys())    # If there's multiple examples

        # Instantiate preprocessing class
        pipeline = RossmanPreprocessing()

        # Data Cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # Feature Engineering
        df2 = pipeline.feature_engineering(df1)

        # Data Preparation
        df3 = pipeline.data_preparation(df2)

        # Prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    else:
        return Response('{}', status='200', mimetype='/application/json')
    
if __name__ == '__main__':
    app.run('0.0.0.0')