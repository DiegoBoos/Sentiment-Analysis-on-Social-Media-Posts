from flask import Flask, render_template, request
import pickle
from transformers import BertTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from emoji import demojize
import re
from transformers import BertTokenizer, BertModel
import numpy as np

app = Flask(__name__)

model = pickle.load(open('./model/lstm_model.pkl', 'rb'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/')
def home():
    return render_template('index.html')

slang_dict = {
    'btw': 'by the way',
    'lol': 'laughing out loud',
    'idk': 'I do not know',
    'imo': 'in my opinion',
    'imho': 'in my humble opinion',
    'brb': 'be right back',
    'tbh': 'to be honest',
    'lmao': 'laughing my ass off',
    'rofl': 'rolling on the floor laughing',
    'smh': 'shaking my head',
    'omg': 'oh my god',
    'ttyl': 'talk to you later',
    'afaik': 'as far as I know',
    'irl': 'in real life',
    'thx': 'thanks',
    'pls': 'please',
    'dm': 'direct message',
    'fyi': 'for your information',
    'b4': 'before',
    'gr8': 'great',
    'u': 'you',
    'r': 'are',
    'yolo': 'you only live once',
    'np': 'no problem',
    'g2g': 'got to go',
    'tldr': 'too long, didn’t read',
    'jk': 'just kidding',
    'bff': 'best friends forever',
    'icymi': 'in case you missed it',
    'fomo': 'fear of missing out',
    'ftw': 'for the win',
    'wtf': 'what the f***',
    'nsfw': 'not safe for work',
    'nbd': 'no big deal',
    'faq': 'frequently asked questions',
    'afk': 'away from keyboard',
    'asap': 'as soon as possible'
}

# Extract and process hashtags
def process_hashtags(text):
    hashtags = re.findall(r'#\w+', text)
    return ' '.join(hashtags)

# Preprocess the input text
def preprocess_text(text):
    # Convert emojis to text and process hashtags
    text = demojize(text) + ' ' + process_hashtags(text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and apply lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower())
              for token in tokens if token.lower() not in stop_words]

    # Translate slang
    tokens = [slang_dict.get(token, token) for token in tokens]

    # Remove non-alphabetic characters and keep the tokens
    tokens = [token for token in tokens if token.isalpha()]

    return ' '.join(tokens)

# Predict the sentiment of the input text
def predict_sentiment(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the BERT model
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Function to encode text using BERT
    def bert_encode(text):
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=512, truncation=True, padding='max_length')
        outputs = bert_model(**inputs)
        # bert model will contain the inputs and the output will be the last hidden state
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    # Reshape the BERT features
    bert_features = bert_encode(preprocessed_text)
    bert_features = np.expand_dims(bert_features, axis=1)

    # Make the prediction using the model
    sentiment_probabilities = model(bert_features)
    predicted_sentiment = np.argmax(sentiment_probabilities, axis=1)
    encoded_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = encoded_labels[predicted_sentiment[0]]

    return predicted_sentiment


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']

        predicted_sentiment = predict_sentiment(sentence)
        
        print(predicted_sentiment)

        return render_template('index.html', prediction=predicted_sentiment)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
