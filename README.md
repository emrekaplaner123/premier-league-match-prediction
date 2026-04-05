# Premier League Match Outcome Prediction

Exploratory machine learning project for predicting Premier League match outcomes from historical match data.

## Overview

This project builds an end-to-end prediction pipeline in Python for classifying Premier League matches as:

- Home win
- Draw
- Away win

The workflow includes data collection, preprocessing, Elo rating generation, feature engineering, model training, evaluation, and prediction.

The main goal of the project was to explore how far simple machine learning methods can go on a noisy real-world forecasting problem, while also building a complete and structured pipeline.

## Project Structure

```text
premier-league-match-prediction/
├── data/
├── src/
├── Stats/
├── Tuning/
├── Tuning_old/
├── tests_old/
├── main.py
├── README.md
└── requirments
```

## Pipeline

The project is organized around main.py and includes the following steps:

collect and combine match data
preprocess and clean the dataset
compute Elo ratings
engineer match-level features
train a machine learning model
evaluate predictive performance
generate match outcome predictions

## Features

The model uses a mix of handcrafted football and context features, including:

Elo-based team strength measures
recent team performance statistics
match-level context variables
rolling historical indicators

These features are designed to capture relative team strength and recent form before each match.

## Model

The project uses a Random Forest classifier as a baseline machine learning model for predicting match outcomes.

This is an exploratory project rather than a production-ready forecasting system, and the results are modest. The main value of the project is in feature construction, pipeline building, and evaluation of a difficult prediction task.
