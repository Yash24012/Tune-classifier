# Tune-classifier
# Music Genre Classification

## Overview

This project implements a music genre classification system that utilizes various machine learning algorithms to classify audio tracks into different genres. The system achieves high accuracy by analyzing both audio features and song lyrics.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Classification of 10 different music genres based on audio features such as rhythm, tempo, and pitch.
- Implementation of multiple machine learning algorithms, including:
  - Logistic Regression
  - Naïve Bayes
  - Linear SVM
  - Polynomial SVM
  - Gaussian SVM
  - K-Nearest Neighbors (KNN)
  - Convolutional Neural Networks (CNN)
- Speech recognition to extract lyrics from songs.
- Analysis of song lyrics using transformer-based models, including BERT, GloVe, and Word2Vec, to classify based on themes and emotions.
- Extensive exploratory data analysis (EDA) and feature engineering to identify key features for classification.
- Comprehensive data preprocessing pipelines for both audio and text data.

## Technologies Used

- Python
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow/Keras (for CNN)
  - NLTK/Transformers (for text analysis)
  - Librosa (for audio feature extraction)
  - SpeechRecognition (for extracting lyrics)

## Data

The dataset used for this project includes audio files and their corresponding genre labels. The audio features were extracted using the Librosa library, and lyrics were obtained through speech recognition.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
