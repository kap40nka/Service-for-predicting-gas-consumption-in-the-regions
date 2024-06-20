from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import io
import base64
from keras.saving import register_keras_serializable
from keras import optimizers
import os
from datetime import datetime

app = Flask(__name__, static_folder='static', static_url_path='')

# Регистрация пользовательской метрики, если она используется
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Загружаем предобученные модели для разных регионов
models = {
    f'{city+1}': tf.keras.models.load_model(f'models/model_city_{city+1}.h5', custom_objects={'mse': mse}) for city in range(63)

}

# Компилируем модели заново после загрузки
for model in models.values():
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

# Загружаем данные для разных регионов
df = pd.read_excel(r'data\gg.xlsx',  sheet_name=["Температура"])
df_holiday = pd.read_excel(r'data\gg.xlsx',  sheet_name=["Праздники"])
df_consumption = pd.read_excel(r'data\gg.xlsx',  sheet_name=['Потребление'])

cities = []
for i in range(63):
  df_i = df['Температура'][['Дата',i]]
  df_i = df_i.merge(df_holiday['Праздники'][['Дата','id']], on='Дата', how='inner')
  df_i = df_i.merge(df_consumption['Потребление'][['Дата',i]], on='Дата', how='inner')
  df_i.rename(columns={f'{i}_x':'Температура',f'{i}_y':'Потребление'}, inplace=True)
  cities.append(df_i)

# Загружаем скейлеры
scalers = {
    f'{city+1}': (pickle.load(open(f'models/scaler_city_{city+1}.pkl', 'rb')), pickle.load(open(f'models/scaler_target_city_{city+1}.pkl', 'rb')))
    for city in range(63)
}

def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(prediction)
    plt.plot(num_in, np.array(history[:, 2]), label='History')
    if prediction.any():
        plt.plot(np.arange(num_out)/1, np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    region = data['region']
    day = data['day']
    month = data['month']
    year =  data['year']
    consumption_rate = data['consumption_rate']
    date_string = f"{year}-{month}-{day}"

    # Выбираем соответствующую модель и данные на основе региона
    model = models.get(region)
    city_data = cities[int(region)-1]
    scaler, scaler_target = scalers.get(region)

    if model is None or city_data is None or scaler is None or scaler_target is None:
        return jsonify({'error': 'Invalid region specified'}), 400

    # Подготовка данных
    features_considered_full = ['Температура', 'id', 'Потребление']
    features_full = city_data[features_considered_full]
    features_full.index = city_data['Дата']
    # Преобразуем строку в объект datetime
    target_date = datetime.strptime(date_string, '%Y-%m-%d')
    index_of_target_date = features_full.index.get_loc(target_date)

    features = features_full.iloc[index_of_target_date - 51 : index_of_target_date + 1]
    
    features[['Температура', 'id']] = scaler.transform(features[['Температура', 'id']])
    features[['Потребление']] = scaler_target.transform(features[['Потребление']])


    latest_data = features.values
    input_data = np.array([latest_data])
    
    # Выполняем предсказание
    prediction = model.predict(input_data)[0]
    
    # Возвращаем предсказанное значение к исходному формату
    prediction_rescaled = scaler_target.inverse_transform(prediction.reshape(-1, 1)).flatten()

    # Построение графика
    buf = multi_step_plot(latest_data, prediction[:int(consumption_rate)])

    # Возвращаем предсказание и график в ответе
    return jsonify({
        'prediction': prediction_rescaled.tolist()[:int(consumption_rate)],
        'plot_url': f'data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}'
    })

if __name__ == '__main__':
    app.run(debug=True)


