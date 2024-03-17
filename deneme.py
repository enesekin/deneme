import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# Bitcoin verilerini anlık olarak çekin
bitcoin_data = yf.download('BTC-USD', start='2020-01-01', end='2024-02-17', progress=False)

# Veriyi işleyin ve ölçeklendirin
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bitcoin_data['Close'].values.reshape(-1, 1))

# Eğitim veri setini oluşturun
train_data = scaled_data[0:int(len(scaled_data)*0.95)]

# Veriyi özellik ve etiketlere ayırın
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

y_train = y_train.reshape(-1, 1)

# LSTM modelini oluşturun
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Modeli derleyin
model.compile(optimizer='adam', loss='mean_squared_error')

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Modeli eğitin
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Anlık veriyi çekin
current_data = yf.download('BTC-USD', start='2022-01-01', end='2024-02-18', progress=False)

# Veriyi ölçeklendirin
last_60_days = current_data['Close'].values[-60:].reshape(-1, 1)
last_60_days_scaled = scaler.transform(last_60_days)

# Test veri setini oluşturun
x_test = np.array([last_60_days_scaled])

# Tahmin yapın
predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)

# Alım/satım sinyallerini oluşturun (örnek olarak basit bir sinyal stratejisi kullanılmıştır)
signal = 1 if predicted_price > np.mean(scaler.inverse_transform(scaled_data)[-60:]) else -1

# Gerçekleşen karı hesaplayın
y_test = current_data['Close'].values[-1]
#profit = (y_test - last_60_days[-1]) * signal[0]

print("Tahmin Edilen Fiyat:", predicted_price[0, 0])
print("Alım/Satım Sinyali:", "Al" if signal == 1 else "Sat")
#print("Toplam Kar/Zarar:", profit)
