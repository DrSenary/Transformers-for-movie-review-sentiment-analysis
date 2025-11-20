# Transformers-for-movie-review-sentiment-analysis
SentimentScope: PyTorch transformer for binary sentiment classification on IMDB reviews. Loads, tokenizes, trains, and tests a custom model for recommendation systems; achieves competitive accuracy for NLP tasks.

SentimentScope: Transformer-Based Sentiment Analysis on IMDB
This project implements a custom transformer neural network for binary sentiment classification of IMDB movie reviews. Developed as part of the Udacity AWS AI Nanodegree, it demonstrates how NLP models can power better recommendation systems by understanding user sentiment.

Overview:
CineScope, an entertainment company, aims to recommend content based on user feedback. The goal is to train a PyTorch-based transformer to classify reviews as positive or negative, improving recommendation accuracy.

Key Components:

Data Preparation: Loads and cleans IMDB reviews, splits into train, validation, and test sets, and prepares labels for binary classification.

Tokenization: Uses Hugging Face’s bert-base-uncased tokenizer for efficient subword tokenization.

Model Architecture: Implements a multi-layer transformer (attention, feed-forward blocks, mean pooling) for sentence-level classification.

Training & Validation: Trains the model over multiple epochs, monitors loss and accuracy, and tunes hyperparameters for optimal results.

Testing: Evaluates final performance on the test set, aiming for greater than 75% accuracy.

Technologies Used:

PyTorch

Hugging Face Transformers

Pandas

Matplotlib

Instructions:

Clone the repository and follow instructions in the notebook.

Download and extract the IMDB dataset.

Run the workflow to preprocess data, train the model, and evaluate accuracy.

Results:
After training and tuning, the transformer model achieves competitive sentiment classification accuracy on movie reviews, demonstrating the power and flexibility of transformer-based approaches for real-world NLP tasks.
