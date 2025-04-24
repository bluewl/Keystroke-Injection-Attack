# train_typenet_triplet_preproc.py

import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict

from keras import layers, models, optimizers, backend as K
from keras.layers import Input, LSTM, BatchNormalization, Dropout, Masking, Lambda
from keras.models import Model
from keras.utils import Sequence

# from tensorflow.keras import layers, models, optimizers, backend as K
# from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dropout, Masking, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import Sequence

SEQ_LENGTH         = 70
NUM_FEATURES       = 5
EMBEDDING_DIM      = 128
SEQUENCES_PER_USER = 15
MARGIN             = 1.5

def extract_sequences(df, max_users):
    """
    Assumes df has columns:
      ['PARTICIPANT_ID','HL','IL','PL','RL','KEYCODE'].
    Treats KEYCODE as already normalized in [0,1].
    Windows each user's data into exactly SEQUENCES_PER_USER
    non-overlapping chunks of length SEQ_LENGTH.
    Returns dict {user_id: np.array((15,70,5))}.
    """
    # rename for consistency
    df = df.rename(columns={"PARTICIPANT_ID":"user_id", "KEYCODE":"keycode_norm"})
    # filter users
    df = df[df["user_id"] < max_users]
    user_seqs = defaultdict(list)
    for uid, g in df.groupby("user_id"):
        feats = g[["HL","IL","PL","RL","keycode_norm"]].values
        for start in range(0, len(feats), SEQ_LENGTH):
            end = start + SEQ_LENGTH
            if end > len(feats):
                break
            user_seqs[uid].append(feats[start:end])
            if len(user_seqs[uid]) >= SEQUENCES_PER_USER:
                break
    # only keep users with full 15 windows
    return {
        uid: np.stack(seq_list)
        for uid, seq_list in user_seqs.items()
        if len(seq_list) == SEQUENCES_PER_USER
    }

def triplet_loss(y_true, y_pred):
    return tf.reduce_mean(tf.maximum(y_pred + MARGIN, 0))

def build_embedding_model():
    inp = Input(shape=(SEQ_LENGTH, NUM_FEATURES))
    x = Masking()(inp)
    x = LSTM(EMBEDDING_DIM, return_sequences=True, recurrent_dropout=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = LSTM(EMBEDDING_DIM, recurrent_dropout=0.2)(x)
    emb = BatchNormalization(name="embedding")(x)
    return Model(inp, emb, name="typenet_embedding")

class TripletGenerator(Sequence):
    def __init__(self, user_seqs, batch_size=512, steps_per_epoch=150, **kwargs):
        super().__init__(**kwargs)
        self.user_ids = list(user_seqs.keys())
        self.user_seqs = user_seqs
        self.batch_size = batch_size
        self.steps = steps_per_epoch

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        A, P, N = [], [], []
        for _ in range(self.batch_size):
            uid = random.choice(self.user_ids)
            seqs = self.user_seqs[uid]
            i, j = random.sample(range(SEQUENCES_PER_USER), 2)
            A.append(seqs[i])
            P.append(seqs[j])
            neg_uid = random.choice(self.user_ids)
            while neg_uid == uid:
                neg_uid = random.choice(self.user_ids)
            N.append(random.choice(self.user_seqs[neg_uid]))
        return (np.stack(A), np.stack(P), np.stack(N)), np.zeros((self.batch_size, 1))

def main(args):
    # load CSV (has PARTICIPANT_ID, HL, IL, PL, RL, KEYCODE)
    df = pd.read_csv(args.train_csv)
    user_seqs = extract_sequences(df, args.num_users)

    # build embedding and triplet model
    embed_model = build_embedding_model()
    a_in = Input((SEQ_LENGTH, NUM_FEATURES), name="anchor")
    p_in = Input((SEQ_LENGTH, NUM_FEATURES), name="positive")
    n_in = Input((SEQ_LENGTH, NUM_FEATURES), name="negative")

    em_a = embed_model(a_in)
    em_p = embed_model(p_in)
    em_n = embed_model(n_in)

    def sq_dist(x, y):
        return tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)

    d_ap = Lambda(lambda z: sq_dist(z[0], z[1]), output_shape=(1,), name="d_ap")([em_a, em_p])
    d_an = Lambda(lambda z: sq_dist(z[0], z[1]), output_shape=(1,), name="d_an")([em_a, em_n])
    diff = Lambda(lambda z: z[0] - z[1], output_shape=(1,), name="diff")([d_ap, d_an])

    model = Model([a_in, p_in, n_in], diff, name="typenet_triplet")
    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        ),
        loss=triplet_loss
    )

    # train with generator
    gen = TripletGenerator(
        user_seqs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch
    )
    model.fit(gen, epochs=args.epochs, steps_per_epoch=len(gen))

    # save the embedding extractor
    #embed_model.save("typenet_triplet_embedding_og.keras")
    embed_model.save(args.embedding_out)
    print(f"Triplet embedding saved to {args.embedding_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",       required=True,
                        help="example1.csv with PARTICIPANT_ID, HL,IL,PL,RL,KEYCODE")
    parser.add_argument("--num_users",       type=int, default=68000,
                        help="number of users in CSV (e.g. 68000)")
    parser.add_argument("--batch_size",      type=int, default=512)
    parser.add_argument("--steps_per_epoch", type=int, default=150)
    parser.add_argument("--epochs",          type=int, default=200)
    parser.add_argument("--embedding_out",   default="typenet_triplet_embedding_lol.keras",
                        help="where to save the embedding model")
    args = parser.parse_args()
    main(args)
[]