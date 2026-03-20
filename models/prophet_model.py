from prophet import Prophet

def train_prophet(data):
    df = data.rename(columns={"date": "ds", "value": "y"})
    model = Prophet()
    model.fit(df)
    return model

def predict_prophet(model, days=7):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]