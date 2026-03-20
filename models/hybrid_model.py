def combine(prophet_pred, lstm_pred):
    return (prophet_pred + lstm_pred) / 2