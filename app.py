from flask import Flask, render_template, request
import pickle
from transformers import BertTokenizer
import torch

app = Flask(__name__)

model = pickle.load(open('./model/lstm_model.pkl', 'rb'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']
        
         # Tokenize and encode the input text using BERT tokenizer
        encoded_input = tokenizer(sentence, return_tensors='pt')
        
       # Extract BERT embeddings for the input
        with torch.no_grad():
            embeddings = model.forward(**encoded_input).last_hidden_state

        # Extract features (e.g., mean pooling) from BERT embeddings
        features = embeddings.mean(dim=1)

        # Combine BERT features with other features if applicable

        # Convert to a NumPy array for compatibility with SVM
        features_np = features.numpy()

        # Predict sentiment using the SVM model
        my_prediction = model.predict(features_np)
        
        return render_template('index.html', prediction=my_prediction)
    else:
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)