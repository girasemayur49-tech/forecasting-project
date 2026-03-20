from fastapi import FastAPI
from utils.preprocess import load_data, prepare_lstm_data
from models.prophet_model import train_prophet, predict_prophet
from models.lstm_model import train_lstm
from models.hybrid_model import combine

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/predict")
def predict():
    df = load_data()

    # Prophet
    prophet_model = train_prophet(df)
    prophet_pred = predict_prophet(prophet_model)

    # LSTM
    X, y = prepare_lstm_data(df)
    lstm_model = train_lstm(X, y)

    # Dummy LSTM prediction
    lstm_pred = prophet_pred['yhat'] * 0.95  

    # 🔥 FINAL HYBRID
    hybrid = combine(prophet_pred['yhat'], lstm_pred)

    return {
        "status": "success",
        "hybrid_output": hybrid.tail(5).tolist()
    }