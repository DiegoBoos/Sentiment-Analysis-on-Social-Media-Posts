from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        if output == 1:
            return render_template('index.html', prediction_text='The patient has a heart disease')
        else:
            return render_template('index.html', prediction_text='The patient does not have a heart disease')
    else:
        return render_template('index.html')