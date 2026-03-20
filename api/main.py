from fastapi import FastAPI
from utils.preprocess import load_data, prepare_lstm_data
from models.prophet_model import train_prophet, predict_prophet
from models.lstm_model import train_lstm

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/predict")
def predict():
    df = load_data()

    prophet_model = train_prophet(df)
    prophet_pred = predict_prophet(prophet_model)

    X, y = prepare_lstm_data(df)
    lstm_model = train_lstm(X, y)

    return {
        "status": "success",
        "prophet_output": prophet_pred.tail(5).to_dict()
    }