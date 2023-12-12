# Social Media Sentiment Analysis System

Social media platforms generate a massive amount of textual data with diverse sentiments. The objective of this project is to develop a sentiment analysis system capable of automatically classifying social media posts into positive, negative, or neutral sentiments.

## 1. Data Collection – (30)
- Gather a dataset of social media posts from one of various sources (Twitter, Reddit, etc.). (20)
    - Facebook provides an API but it is complicated. You can find it [here](#).
    - You can make use of reddit API (PRAW) [here](#).
    - Or you can make use of a Web Scraping techniques like Selenium, or BeautifulSoap libraries.
- Annotate the dataset with sentiment labels (positive, negative, neutral) based on the post's sentiment. (10)
    - You can use pretrained models from [Hugging Face](https://huggingface.co/inference-api) to annotate your dataset.

## 2. Preprocessing (20)
- Perform necessary text preprocessing steps such as tokenization, stop-word removal, stemming/lemmatization, and lowercasing. (10)
- Handle specific challenges of social media text like hashtags, emojis, and slang. (10)

## 3. Feature Extraction (20)
- Explore different feature representation methods such as bag-of-words, TF-IDF, word embeddings (e.g., Word2Vec or GloVe), or contextual embeddings (e.g., BERT or GPT). Experiment with 3 different feature extraction techniques to capture meaningful representations of social media text.

## 4. Model Selection and Training (10)
- Choose a suitable machine learning algorithm (e.g., Naive Bayes, SVM, or neural networks) or deep learning model for sentiment classification. 
- Split the dataset into training and testing sets.
- Train the selected model using the training data, evaluate and record its performance on the training and testing data.

## 5. Deployment and Interface (10)
- Develop a simple user-friendly interface using any library (Tkinter is simple one for instance) that allows users to input social media post and obtain sentiment analysis results in real-time.
- Display the sentiment analysis result on the interface and allow user to input additional posts.

## 6. Documentation and Presentation (10)
- Create a comprehensive report documenting the project's methodology, results, and findings.
- Prepare a presentation to showcase the sentiment analysis system, discuss challenges faced, and highlight insights gained from the project. Including a live demonstration test cases that will be tested during the presentation which will be handled In Class.