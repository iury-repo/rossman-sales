import os
import pandas as pd
import math
import numpy as np
import json
import requests

df_load = pd.read_csv('/home/iury_unix/ml_projects/rossman_sales_prediction/data/raw/test.csv')
df_store = pd.read_csv('/home/iury_unix/ml_projects/rossman_sales_prediction/data/raw/store.csv')

# Merge test and store
df_test = pd.merge(df_load, df_store, how='left', on='Store')

# Choosing one specific store to test
df_test = df_test[df_test['Store'] == 22]

# Removing unused columns
df_test = df_test[df_test['Open'] != 0]
df_test = df_test[~df_test['Open'].isnull()]
df_test = df_test.drop('Id', axis=1)

# Converting to json
data = json.dumps(df_test.to_dict(orient='records'))

# API call
url = 'http://0.0.0.0:5000/rossman/predict'
header = {'Content-type': 'application/json'}
data = data

r = requests.post(url=url, data=data, headers=header)
print(f'Status Code: {r.status_code}')