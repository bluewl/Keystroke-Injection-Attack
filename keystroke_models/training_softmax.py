import pandas as pd
import numpy as np
from keras import layers, models, optimizers
from sklearn.preprocessing import LabelEncoder


SEQ_LENGTH = 70
NUM_FEATURES = 5
NUM_USERS = 10000
SEQUENCES_PER_USER = 15
NUM_CLASSES = NUM_USERS
EMBEDDING_DIM = 128
BATCH_SIZE = 512
EPOCHS = 200

def extract_sequences(df, preprocessed=False):
    """
    If preprocessed=True, df must already have columns:
      ['user_id','press_time','HL','IL','PL','RL','keycode_norm'] 
    where HL/IL/PL/RL are in seconds and keycode_norm in [0,1].
    Otherwise, this will compute them from raw press/release times.
    """
    sequences, labels = [], []
    

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("PRESS_TIME")
        if not preprocessed:
            # Raw timing features (timestamps in ms)
            group["HL"] = (group["RELEASE_TIME"] - group["PRESS_TIME"]) / 1000.0
            group["IL"] = (group["PRESS_TIME"].shift(-1) - group["RELEASE_TIME"]) / 1000.0
            group["PL"] = (group["PRESS_TIME"].diff()) / 1000.0
            group["RL"] = (group["RELEASE_TIME"].diff()) / 1000.0
            # Normalize keycode
            group["keycode_norm"] = group["KEYCODE"] / 255.0
            # Fill NaNs (first/last diffs) with zeros
            group.fillna(0, inplace=True)
        
        # Window into fixed-length SEQ_LENGTH chunks
        group = group.rename(columns={"KEYCODE":"keycode_norm"})
        feats = group[["HL","IL","PL","RL","keycode_norm"]].values
        # non-overlapping windows
        for start in range(0, len(feats), SEQ_LENGTH):
            end = start + SEQ_LENGTH
            if end > len(feats):
                break
            sequences.append(feats[start:end])
            labels.append(user_id)
            if len(labels) >= SEQUENCES_PER_USER * (user_id+1):  
                break

    return np.array(sequences), np.array(labels)



df = pd.read_csv("../first_10k_participants_normalized_padded.csv")
df = df.rename(columns={"PARTICIPANT_ID":"user_id"})
df = df[df["user_id"] < NUM_USERS]
#X, y = extract_sequences(df, preprocessed=False)
X, y = extract_sequences(df, preprocessed=True)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

def build_typenet_softmax():
    inp = layers.Input(shape=(SEQ_LENGTH, NUM_FEATURES))
    x = layers.Masking()(inp)
    x = layers.LSTM(EMBEDDING_DIM, return_sequences=True, recurrent_dropout=0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(EMBEDDING_DIM, recurrent_dropout=0.2)(x)
    emb = layers.BatchNormalization(name="embedding")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_output")(emb)
    return models.Model(inp, out)

model = build_typenet_softmax()
model.compile(
    optimizer=optimizers.Adam(
        learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X, y_encoded,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    validation_split=0.1
)


feature_model = models.Model(
    inputs=model.input,
    outputs=model.get_layer("embedding").output
)
feature_model.save("typenet_embedding_model.keras")