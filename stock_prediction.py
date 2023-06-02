import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


tf.random.set_seed(42)


def generate_features(df):

    df_new = pd.DataFrame()

    # convert them to float
    df['Close'] = df['Close'].str.replace(',', '')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = df['Open'].str.replace(',', '')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = df['High'].str.replace(',', '')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = df['Low'].str.replace(',', '')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = df['Volume'].str.replace(',', '')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)

    # average price
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    # average volume
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']
    # standard deviation of prices
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']
    # standard deviation of volumes
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']
    # # return
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)
    # the target
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


data_raw = pd.read_csv('data.csv', index_col='Date')
data = generate_features(data_raw)


start_train = 0
end_train = 5700

start_test = 5701
end_test = 6796

train_size = int(len(data) * 0.8)  # 80% for training, 20% for testing

data_train = data.iloc[:train_size]
X_train = data_train.drop('close', axis=1).values
y_train = data_train['close'].values

data_test = data.iloc[train_size:]
X_test = data_test.drop('close', axis=1).values
y_test = data_test['close'].values

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

learning_rate = 0.3
epocs_num = 1000
units= 64

model = Sequential([
    Dense(units=units, activation='relu'),
    Dense(units=1)
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

model.fit(X_scaled_train, y_train, epochs=epocs_num, verbose=True)

predictions = model.predict(X_scaled_test)[:, 0]

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print(f'MSE: {mean_squared_error(y_test, predictions):.3f}')
print(f'MAE: {mean_absolute_error(y_test, predictions):.3f}')
print(f'R^2: {r2_score(y_test, predictions):.3f}')

# from tensorboard.plugins.hparams import api as hp
# HP_HIDDEN = hp.HParam('hidden_size', hp.Discrete([64, 32, 16]))
# HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300, 1000]))
# HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.01, 0.4))
#
#
# def train_test_model(hparams, logdir):
#     model = Sequential([
#         Dense(units=hparams[HP_HIDDEN], activation='relu'),
#         Dense(units=1)
#     ])
#     model.compile(loss='mean_squared_error',
#                   optimizer=tf.keras.optimizers.Adam(hparams[HP_LEARNING_RATE]),
#                   metrics=['mean_squared_error'])
#     model.fit(X_scaled_train, y_train, validation_data=(X_scaled_test, y_test), epochs=hparams[HP_EPOCHS], verbose=False,
#               callbacks=[
#                   tf.keras.callbacks.TensorBoard(logdir),  # log metrics
#                   hp.KerasCallback(logdir, hparams),  # log hparams
#                   tf.keras.callbacks.EarlyStopping(
#                       monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='auto',
#                   )
#               ],
#               )
#     _, mse = model.evaluate(X_scaled_test, y_test)
#     pred = model.predict(X_scaled_test)
#     r2 = r2_score(y_test, pred)
#     return mse, r2
#
# def run(hparams, logdir):
#     with tf.summary.create_file_writer(logdir).as_default():
#         hp.hparams_config(
#             hparams=[HP_HIDDEN, HP_EPOCHS, HP_LEARNING_RATE],
#             metrics=[hp.Metric('mean_squared_error', display_name='mse'),
#                      hp.Metric('r2', display_name='r2')],
#         )
#         mse, r2 = train_test_model(hparams, logdir)
#         tf.summary.scalar('mean_squared_error', mse, step=1)
#         tf.summary.scalar('r2', r2, step=1)
#
#
# session_num = 0
#
# for hidden in HP_HIDDEN.domain.values:
#     for epochs in HP_EPOCHS.domain.values:
#         for learning_rate in tf.linspace(HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value, 5):
#             hparams = {
#                 HP_HIDDEN: hidden,
#                 HP_EPOCHS: epochs,
#                 HP_LEARNING_RATE: float("%.2f"%float(learning_rate)),
#             }
#             run_name = "run-%d" % session_num
#             print('--- Starting trial: %s' % run_name)
#             print({h.name: hparams[h] for h in hparams})
#             run(hparams, 'logs/hparam_tuning/' + run_name)
#             session_num += 1
#
#
#
# model = Sequential([
#     Dense(units=16, activation='relu'),
#     Dense(units=1)
# ])
#
# model.compile(loss='mean_squared_error',
#               optimizer=tf.keras.optimizers.Adam(0.21))
#
# model.fit(X_scaled_train, y_train, epochs=1000, verbose=False)
#
# predictions = model.predict(X_scaled_test)[:, 0]

plt.plot(data_test.index, y_test, c='k')
plt.plot(data_test.index, predictions, c='b')
plt.plot(data_test.index, predictions, c='r')
plt.plot(data_test.index, predictions, c='g')
plt.xticks(range(0, 252, 10), rotation=60)
plt.xlabel('Date')
plt.ylabel('Close price')
plt.legend(['Truth', 'Neural network prediction'])
plt.show()
