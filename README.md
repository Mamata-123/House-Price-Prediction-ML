
# House Price Prediction ML

A Machine Learning project that predicts house prices using the Ames, Iowa housing dataset.

## Features
- Loads housing data using Pandas
- Trains a Random Forest Regression model
- Evaluates the model using MAE and R2 score
- Predicts estimated house prices through a Streamlit web app

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib

## Run the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the web app:

```bash
python -m streamlit run app.py
```

## Dataset
House Prices – Advanced Regression Techniques, Kaggle.

Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

## Limitation
This model uses housing data from Ames, Iowa, USA. Predictions are educational estimates and are not reliable estimates of Nepal's housing market.
