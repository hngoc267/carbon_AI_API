import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input
from sklearn.preprocessing import MinMaxScaler
import os

# --- LƯU Ý: Bạn cần cài thư viện openpyxl để pandas đọc được file Excel ---
# Lệnh cài: pip install openpyxl

# 1. ĐỌC DỮ LIỆU THỰC TẾ TỪ FILE EXCEL
file_path = 'carbon_data.xlsx'  # Đổi từ .csv sang .xlsx
try:
    # Sử dụng read_excel thay vì read_csv
    df = pd.read_excel(file_path)

    # In ra 5 dòng đầu để kiểm tra xem đã đọc đúng chưa
    print("Dữ liệu vừa đọc từ Excel:")
    print(df.head())

    # Kiểm tra xem cột 'carbon_actual' có tồn tại không
    if 'carbon_actual' not in df.columns:
        print(f"LỖI: Không tìm thấy cột 'carbon_actual'. Các cột hiện có là: {df.columns.tolist()}")
        exit()

except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    exit()

# Chỉ lấy cột carbon_actual để huấn luyện
data = df['carbon_actual'].values.reshape(-1, 1)

# 2. CHUẨN HÓA DỮ LIỆU (Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. TẠO CẤU TRÚC DỮ LIỆU CHUỖI THỜI GIAN
X, y = [], []
look_back = 24
if len(scaled_data) <= look_back:
    print(f"Dữ liệu quá ít ({len(scaled_data)} dòng)! Cần ít nhất 25 dòng.")
else:
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 4. XÂY DỰNG MÔ HÌNH Bi-LSTM
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        Bidirectional(LSTM(units=64, return_sequences=True)),
        Bidirectional(LSTM(units=32)),
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 5. HUẤN LUYỆN (Training)
    print(f"Bắt đầu huấn luyện với {len(X)} mẫu...")
    model.fit(X, y, batch_size=16, epochs=20, verbose=1)

    # 6. LƯU MÔ HÌNH THÀNH FILE .H5
    model_path = 'bilstm_carbon_model.h5'
    model.save(model_path)

    print("-" * 30)
    print(f"THÀNH CÔNG! Đã tạo file mô hình: {os.path.abspath(model_path)}")