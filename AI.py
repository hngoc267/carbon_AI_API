from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Chỉ import những thứ thực sự cần thiết để tiết kiệm RAM
import tensorflow as tf 
import uvicorn

app = FastAPI()

# Khai báo biến model toàn cục nhưng chưa nạp ngay
model_bilstm = None

def get_model():
    global model_bilstm
    if model_bilstm is None:
        # Nạp mô hình một cách cẩn thận
        model_bilstm = tf.keras.models.load_model('bilstm_carbon_model.h5')
    return model_bilstm

class CarbonRequest(BaseModel):
    data: list

@app.get("/")
async def root():
    return {"status": "AI Server is running"}

@app.post("/predict/sarima")
async def predict_sarima(request: CarbonRequest):
    # ... giữ nguyên code SARIMA cũ ...
    return {"pred_sarima": [round(float(x), 2) for x in forecast]}

@app.post("/predict/bilstm")
async def predict_bilstm(request: CarbonRequest):
    try:
        model = get_model() # Chỉ nạp khi thực sự có yêu cầu gọi đến
        # ... giữ nguyên code xử lý Bi-LSTM cũ ...
        return {"pred_bilstm": final_preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ép dùng cổng từ biến môi trường của Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
