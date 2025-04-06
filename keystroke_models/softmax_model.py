import numpy as np
import pandas as pd
import keras
from keras import layers
import tensorflow as tf
import datetime
import pytz

# ==== Parameters ====
seq_len = 70
num_features = 5
input_shape = (seq_len, num_features)
csv_file_path = "first_10k_participants_normalized_padded.csv"  # Replace with your real file path


# ==== Load and Process CSV ====
df = pd.read_csv(csv_file_path)
feature_cols = ['HL', 'IL', 'PL', 'RL', 'KEYCODE']
grouped = df.groupby(['PARTICIPANT_ID', 'TEST_SECTION_ID'])

x_data, y_labels = [], []

for (pid, _), group in grouped:
    features = group[feature_cols].values
    if len(features) < 10:
        continue
    if len(features) < seq_len:
        pad = np.full((seq_len - len(features), len(feature_cols)), -1.0)
        features = np.vstack([features, pad])
    else:
        features = features[:seq_len]
    x_data.append(features)
    y_labels.append(pid)

x_data = np.array(x_data)
y_labels = np.array(y_labels)

# Encode user IDs to integers and then to one-hot vectors
unique_ids = np.unique(y_labels)
id_to_index = {uid: i for i, uid in enumerate(unique_ids)}
y_indices = np.array([id_to_index[uid] for uid in y_labels])
y_data = tf.keras.utils.to_categorical(y_indices, num_classes=len(unique_ids))


# ==== Define Model ====
def create_LSTM_softmax_network(input_shape, num_users):
    input = layers.Input(shape=input_shape)
    masked = layers.Masking(mask_value=-1.0)(input)
    normalized_masked = layers.BatchNormalization()(masked)
    first_LSTM = layers.LSTM(units=128, activation="tanh",
                              recurrent_activation="sigmoid", dropout=0.5,
                              recurrent_dropout=0.2, return_sequences=True)(normalized_masked)
    normalized_first_LSTM = layers.BatchNormalization()(first_LSTM)
    second_LSTM = layers.LSTM(units=128, activation="tanh",
                               recurrent_activation="sigmoid", recurrent_dropout=0.2)(normalized_first_LSTM)
    outputs = layers.Dense(units=num_users, activation="softmax")(second_LSTM)
    return keras.Model(input, outputs)


# ==== Compile and Train ====
model = create_LSTM_softmax_network(input_shape=input_shape, num_users=len(unique_ids))
optimizer = keras.optimizers.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=200, steps_per_epoch=150, batch_size=512)  # adjust epochs/batch as needed

# ==== Save Model ====
model.save("softmax_typenet_model.keras")

# ==== Timestamp ====
eastern = pytz.timezone('US/Eastern')
now = datetime.datetime.now(eastern)
print("Model saved at:", now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
