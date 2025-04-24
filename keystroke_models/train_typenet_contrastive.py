# train_typenet_contrastive.py

import argparse
import random
import numpy as np
import pandas as pd
from collections import defaultdict

import tensorflow as tf
from keras import layers, models, optimizers, backend as K
from keras.layers import Input, LSTM, BatchNormalization, Dropout, Masking, Lambda
from keras.models import Model

SEQ_LENGTH       = 70
NUM_FEATURES     = 5
EMBEDDING_DIM    = 128
SEQUENCES_PER_USER = 15
MARGIN           = 1.5

def extract_sequences(df):
    """
    Assumes df has columns ['user_id','HL','IL','PL','RL','keycode_norm'].
    Returns:
      user_seqs: {user_id: np.array([seq1,seq2,...])} where each seq is (70×5)
    """
    user_seqs = defaultdict(list)
    df = df.rename(columns={"KEYCODE":"keycode_norm"})
    for uid, g in df.groupby("user_id"):
        feats = g[["HL","IL","PL","RL","keycode_norm"]].values
        # non-overlapping windows of length SEQ_LENGTH
        for start in range(0, len(feats), SEQ_LENGTH):
            end = start + SEQ_LENGTH
            if end > len(feats):
                break
            user_seqs[uid].append(feats[start:end])
            if len(user_seqs[uid]) >= SEQUENCES_PER_USER:
                break
    # to numpy arrays
    return {uid: np.stack(seq_list) 
            for uid, seq_list in user_seqs.items()
            if len(seq_list) == SEQUENCES_PER_USER}

def contrastive_loss(y_true, y_pred):
    # y_true: 0 = genuine, 1 = impostor; y_pred = d(x1,x2)
    square_pred = tf.square(y_pred)
    margin_dist = tf.square(tf.maximum(MARGIN - y_pred, 0))
    return tf.reduce_mean((1 - y_true) * square_pred + y_true * margin_dist)

def build_base_network():
    inp = Input(shape=(SEQ_LENGTH, NUM_FEATURES))
    x = Masking()(inp)
    x = LSTM(EMBEDDING_DIM, return_sequences=True, recurrent_dropout=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = LSTM(EMBEDDING_DIM, recurrent_dropout=0.2)(x)
    emb = BatchNormalization(name="embedding")(x)
    return Model(inp, emb, name="typenet_base")

class PairGenerator(tf.keras.utils.Sequence):
    def __init__(self, user_seqs, batch_size=512, steps_per_epoch=150,**kwargs):
        super().__init__(**kwargs)
        self.user_ids = list(user_seqs.keys())
        self.user_seqs = user_seqs
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.steps = steps_per_epoch

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        A, B, labels = [], [], []
        # genuine
        for _ in range(self.half):
            uid = random.choice(self.user_ids)
            seqs = self.user_seqs[uid]
            i, j = random.sample(range(SEQUENCES_PER_USER), 2)
            A.append(seqs[i]); B.append(seqs[j]); labels.append(0)
        # impostor
        for _ in range(self.half):
            u1, u2 = random.sample(self.user_ids, 2)
            A.append(random.choice(self.user_seqs[u1]))
            B.append(random.choice(self.user_seqs[u2]))
            labels.append(1)
        A = np.stack(A); B = np.stack(B); labels = np.array(labels)
        # shuffle
        idxs = np.arange(self.batch_size)
        np.random.shuffle(idxs)
        return (A[idxs], B[idxs]), labels[idxs]

def main(args):
    # load and filter
    df = pd.read_csv(args.train_csv)
    df = df.rename(columns={"PARTICIPANT_ID":"user_id"})
    df = df[df["user_id"] < args.num_users]
    user_seqs = extract_sequences(df)

    # build models
    base_net = build_base_network()
    inp_a = Input((SEQ_LENGTH, NUM_FEATURES))
    inp_b = Input((SEQ_LENGTH, NUM_FEATURES))
    emb_a = base_net(inp_a)
    emb_b = base_net(inp_b)
    # euclidean distance
    dist = Lambda(
    lambda t: tf.sqrt(tf.reduce_sum(tf.square(t[0] - t[1]), axis=1, keepdims=True)),
    output_shape=(1,)
    )([emb_a, emb_b])
    siamese = Model([inp_a, inp_b], dist, name="typenet_siamese")
    siamese.compile(
        optimizer=optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=contrastive_loss
    )

    # data generator
    gen = PairGenerator(user_seqs,
                        batch_size=args.batch_size,
                        steps_per_epoch=args.steps_per_epoch)

    # train
    siamese.fit(
        gen,
        epochs=args.epochs
    )

    # save embedding
    base_net.save(args.embedding_out)
    print(f"Saved embedding model to {args.embedding_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",      required=True,
                        help="preprocessed CSV with HL,IL,PL,RL,keycode_norm")
    parser.add_argument("--num_users",      type=int, default=68000,
                        help="number of users to load (e.g. 68000)")
    parser.add_argument("--batch_size",     type=int, default=512)
    parser.add_argument("--steps_per_epoch",type=int, default=150)
    parser.add_argument("--epochs",         type=int, default=200)
    parser.add_argument("--embedding_out",  default="typenet_contrastive_embedding.keras",
                        help="where to save the trained embedding model")
    args = parser.parse_args()
    main(args)
