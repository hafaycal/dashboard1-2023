from flask import Flask
import pandas as pd
import sklearn
import joblib
from flask import Flask, jsonify, request
import json
from treeinterpreter import treeinterpreter as ti

data_original = pd.read_csv("data/data_original.csv", index_col='SK_ID_CURR')


app = Flask(__name__)

@app.route('/')
def index():
    return 'Web App with Python Flask!'

@app.route('/api/sk_ids/')
# Test : http://127.0.0.1:5000/api/sk_ids/
def sk_ids():
    # Extract list of 'SK_ID_CURR' from the DataFrame
    sk_ids = list(data_original.index)[:50]

    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': sk_ids
     })


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)



