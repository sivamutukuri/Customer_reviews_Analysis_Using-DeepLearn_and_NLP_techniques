# Sentiment Analysis of User Reviews

## Introduction

This project aims to perform sentiment analysis on user reviews to predict whether a user would recommend a product or not (`user_suggestion`). We utilize Natural Language Processing (NLP) techniques and machine learning models to achieve this. The project involves data preprocessing, feature extraction, model training, and evaluation.

## Exploratory Data Analysis (EDA)

1. **Data Loading and Inspection:** The project uses three datasets: `train.csv`, `test.csv`, and `validation.csv`. We load these datasets using pandas and inspect their shapes and head to get a basic understanding of the data.

2. **Data Preprocessing:** We perform the following preprocessing steps:
    - **Tokenization and Lemmatization:** We use spaCy to tokenize the user reviews and lemmatize the words to their base forms. This helps reduce the dimensionality of the data and improve model performance.
    - **Stop Word Removal:** We remove common stop words that do not carry much meaning.
    - **Noun Removal (Optional):** We experiment with removing nouns from the text data to see if it improves model performance. This is done using spaCy's part-of-speech tagging.

3. **Feature Extraction:** We utilize different feature extraction techniques to represent the text data numerically:
    - **Count Vectorization:** We use `CountVectorizer` to convert text into a matrix of token counts.
    - **TF-IDF Vectorization:** We use `TfidfVectorizer` to create a matrix of TF-IDF values, which considers the importance of words in the entire corpus.
    - **N-grams:** We experiment with using unigrams, bigrams, and trigrams as features to capture more context from the text data.

## Models and Techniques

We experimented with several machine learning models and techniques:

1. **Naive Bayes:** We use `MultinomialNB` as a baseline model due to its simplicity and effectiveness for text classification tasks.

2. **Neural Networks:** We implement a simple feedforward neural network with batch normalization and dropout layers to enhance performance. We use `BCELoss` (Binary Cross-Entropy Loss) as the loss function and `Adam` optimizer for training.

## Results and Evaluation

We evaluate the models using accuracy and F1-score metrics on the validation set. The results show that the Naive Bayes model with TF-IDF vectorization and n-grams performs better than other models. We also observed that adding noun removal step helps to improve the model performance.

## Conclusion

This project demonstrates the application of NLP techniques and machine learning models for sentiment analysis of user reviews. The results show that a relatively simple model like Naive Bayes with carefully chosen features can achieve satisfactory results. Further improvements can be explored by using more complex models or fine-tuning the hyperparameters of the existing models.

## Future Work

- **Explore other machine learning models:** such as logistic regression, support vector machines, and deep learning models.
- **Experiment with different feature engineering techniques:** such as sentiment lexicons, word embeddings, and topic modeling.
- **Fine-tune hyperparameters:** of the models to further improve performance.
- **Deploy the model:** to a real-world application for predicting user sentiment.
