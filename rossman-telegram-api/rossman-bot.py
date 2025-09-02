import os
import pandas as pd
import math
import numpy as np
import json
import requests
from flask import Flask, request, Response 

# constants
TOKEN = '7343342169:AAEbj6Rr2pMyQgUIUlu2JHcGV49N1n5-EAI'

# # Bot info
# https://api.telegram.org/bot7343342169:AAEbj6Rr2pMyQgUIUlu2JHcGV49N1n5-EAI/getMe

# # Get updates
# https://api.telegram.org/bot7343342169:AAEbj6Rr2pMyQgUIUlu2JHcGV49N1n5-EAI/getUpdates

# Set Webhook
# https://api.telegram.org/bot7343342169:AAEbj6Rr2pMyQgUIUlu2JHcGV49N1n5-EAI/setWebhook?url=https://358ead7cb4fc4a.lhr.life

# Webhook Render app
# https://api.telegram.org/bot7343342169:AAEbj6Rr2pMyQgUIUlu2JHcGV49N1n5-EAI/setWebhook?url=https://rossman-bot-6maz.onrender.com


# # Send Message
# https://api.telegram.org/bot7343342169:AAEbj6Rr2pMyQgUIUlu2JHcGV49N1n5-EAI/sendMessage?chat_id=7927049424&text=Hi Iury im doing good! thanks


def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
    url = url + 'sendMessage?chat_id={}'.format(chat_id)

    r= requests.post(url, json={'text': text})
    print(f'Status Code {r.status_code}')

    return None

def load_dataset(store_id):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    df_load = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
    df_store = pd.read_csv(os.path.join(BASE_DIR, 'store.csv'))

    # Merge test and store
    df_test = pd.merge(df_load, df_store, how='left', on='Store')

    # Choosing one specific store to test
    df_test = df_test[df_test['Store'] == store_id]

    if not df_test.empty:        
        # Removing unused columns
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]
        df_test = df_test.drop('Id', axis=1)

        # Converting to json
        data = json.dumps(df_test.to_dict(orient='records'))
    else:
        data = 'error'

    return data

# API call
def predict(data):
    url = 'https://rossman-api-1wo5.onrender.com/rossman/predict'
    header = {'Content-type': 'application/json'}
    data = data

    r = requests.post(url=url, data=data, headers=header)
    print(f'Status Code: {r.status_code}')

    d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())

    return d1

def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '')

    try:
        store_id = int(store_id)

    except ValueError:
        store_id = 'error'

    return chat_id, store_id


# API Initialization
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.get_json()

        chat_id, store_id = parse_message(message)
        if store_id != 'error':
            #loading
            data = load_dataset(store_id)
            if data != 'error':
                #predict
                d1 = predict(data)
            else:
                send_message(chat_id, 'Store not available')
                return Response('Ok', status=200)
            
            #calculation
            d2 = d1[['store', 'predictions']].groupby('store').sum().reset_index()

            #send message
            msg = 'Store Number {} will sell R$ {:,.2f} in the next 6 weeks.'.format(d2['store'].values[0], 
                                                                                     d2['predictions'].values[0])
            send_message(chat_id, msg)
            return Response('Ok', status=200)          
        
        else:
            send_message(chat_id, 'Store ID is Wrong')
            return Response('Ok', status=200)

    else:
        return '<h1> Rossman Telegram Bot <h1>'

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host= '0.0.0.0', port=port)

