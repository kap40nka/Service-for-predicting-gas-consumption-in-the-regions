import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from keras import Sequential, layers, callbacks
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import pickle

# Загрузка данных
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

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plt.plot(num_in, np.array(history[:, 2]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

# Параметры
TRAIN_SPLIT = 3218
future_target = 5
past_history = 52
STEP = 1
BATCH_SIZE = 256
BUFFER_SIZE = 400
EVALUATION_INTERVAL = 200
EPOCHS = 50

for i, city in enumerate(cities):
    features_considered = ['Температура', 'id', 'Потребление']
    features = city[features_considered]
    features.index = city['Дата']
    
    scaler = MinMaxScaler()
    scaler_target = MinMaxScaler()
    features[['Температура', 'id']] = scaler.fit_transform(features[['Температура', 'id']])
    features[['Потребление']] = scaler_target.fit_transform(features[['Потребление']])
    
    dataset = features.values
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 2], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 2],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)
    
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
    
    multi_step_model = Sequential()
    multi_step_model.add(layers.LSTM(32, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(layers.LSTM(16, activation='relu'))
    multi_step_model.add(layers.Dense(future_target))
    optimizer = Adam(learning_rate=0.001)
    
    multi_step_model.compile(optimizer=optimizer, loss='mse')
    
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.00001,  # минимальное улучшение
        patience=5,  # сколько эпох ждать
        restore_best_weights=True,
    )
    
    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_multi,
                                              callbacks=[early_stopping],
                                              validation_steps=50)
    
    plot_train_history(multi_step_history, f'City {i+1} - Multi-Step Training and validation loss')
    
    # Сохранение модели
    model_filename = f'models/model_city_{i+1}.h5'
    multi_step_model.save(model_filename)
    
    # Сохранение скейлеров
    with open(f'models/scaler_city_{i+1}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'models/scaler_target_city_{i+1}.pkl', 'wb') as f:
        pickle.dump(scaler_target, f)
    
    # Выводим предсказания
    val_data_multi_no_repeat = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi)).batch(BATCH_SIZE)
    steps = len(list(val_data_multi_no_repeat))
    predictions = multi_step_model.predict(val_data_multi, steps=steps)
    
    mse_value = 0
    for j in range(len(predictions)):
        mse_value += mean_squared_error(scaler_target.inverse_transform(y_val_multi[j].reshape(-1, 1)), 
                                        scaler_target.inverse_transform(predictions[j].reshape(-1, 1)))
    print(f'City {i+1} - MSE: {mse_value / len(predictions)}')
    
    # Показать несколько предсказаний
    #for x, y in val_data_multi.take(3):
    #    print(multi_step_model.predict(x)[0])
    #    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
