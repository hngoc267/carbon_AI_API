from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import uvicorn

app = FastAPI(title="Carbon Forecasting Comparison API")


class CarbonRequest(BaseModel):
    data: list


# Tải mô hình Bi-LSTM đã train
try:
    model_bilstm = load_model('bilstm_carbon_model.h5')
    print("Bi-LSTM Model Loaded.")
except:
    model_bilstm = None


# --- ENDPOINT 1: SARIMA ---
@app.post("/predict/sarima")
async def predict_sarima(request: CarbonRequest):
    try:
        df = pd.DataFrame(request.data)
        actual_values = pd.to_numeric(df['carbon_actual'], errors='coerce').ffill().values

        model = SARIMAX(actual_values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=48)

        return {"pred_sarima": [round(float(x), 2) for x in forecast]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 2: Bi-LSTM ---
@app.post("/predict/bilstm")
async def predict_bilstm(request: CarbonRequest):
    try:
        if not model_bilstm: raise Exception("Model not found")
        df = pd.DataFrame(request.data)
        actual_values = pd.to_numeric(df['carbon_actual'], errors='coerce').ffill().values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(actual_values.reshape(-1, 1))

        input_seq = scaled_data[-24:].reshape(1, 24, 1)
        predictions = []

        for _ in range(48):
            pred = model_bilstm.predict(input_seq, verbose=0)
            predictions.append(pred[0, 0])
            input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        final_preds = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return {"pred_bilstm": [round(float(x), 2) for x in final_preds]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

