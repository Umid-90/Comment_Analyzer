from flask import Flask, render_template, url_for, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import joblib


app = Flask(__name__, template_folder='template')

# Load the TF-IDF vocabulary
with open(r"tf_idf.joblib", "rb") as f:
    tox = joblib.load(f)
# Load the pickled RDF models
with open(r"Toks.joblib", "rb") as f:
    tox_model = joblib.load(f)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    data = [user_input]
    arr = tox.transform(data)
    pred_tox = tox_model.predict_proba(arr)[:,1]
    output = round(pred_tox[0], 2)
    print(output)
    return render_template('index.html',
                            pred_tox = ' Токсичность текста: {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)
