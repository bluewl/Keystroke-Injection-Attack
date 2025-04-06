import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import random

import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import keras
from keras import layers
from tqdm import tqdm



def create_LSTM_network(input_shape):
    input = layers.Input(shape=input_shape)
    #input = layers.Input(shape=(input_shape,5))
    #masked = layers.Masking(mask_value=-1.0)(input)
    masked = layers.Masking(mask_value=-1.0)(input)
    normalized_masked = layers.BatchNormalization()(masked)
    first_LSTM = layers.LSTM(units=128,activation="tanh",
    recurrent_activation="sigmoid",dropout=0.5,recurrent_dropout=0.2,return_sequences=True)(normalized_masked)

    normalized_first_LSTM = layers.BatchNormalization()(first_LSTM)
    second_LSTM = layers.LSTM(units=128,activation="tanh",
    recurrent_activation="sigmoid",recurrent_dropout=0.2)(normalized_first_LSTM)
    return keras.Model(input, second_LSTM)


#if we pad, we'd have a list of S sequences, each represented by an M*5 matrix
#what needs to be padded is the M dimension S*M*S
#pad with -1 just to be safe
#keras.layers.Masking(mask_value=-1.0, **kwargs)
#padded_sequences = pad_sequences(sequences, padding='post', dtype='float32', value=0)


def create_LSTM_softmax_network(input_shape,num_users):
    #input = layers.Input(shape=input_shape)
    #input = layers.Input(shape=(input_shape,5))
    input = layers.Input(shape=input_shape)
    masked = layers.Masking(mask_value=-1.0)(input)
    normalized_masked = layers.BatchNormalization()(masked)
    first_LSTM = layers.LSTM(units=128,activation="tanh",
    recurrent_activation="sigmoid",dropout=0.5,recurrent_dropout=0.2,return_sequences=True)(normalized_masked)

    normalized_first_LSTM = layers.BatchNormalization()(first_LSTM)
    second_LSTM = layers.LSTM(units=128,activation="tanh",
    recurrent_activation="sigmoid",recurrent_dropout=0.2)(normalized_first_LSTM)
    #logits = layers.Dense(units=num_users,activation="relu")(second_LSTM)
    outputs = layers.Dense(units=num_users,activation="softmax")(second_LSTM)
    return keras.Model(input, outputs)

def contrastive_loss(margin=1.0):
    #y_true is 1 if it's different,y_pred is difference between embedding vectors
    def loss(y_true, y_pred):
        # Compute the Euclidean distance between embeddings
        y_true = keras.ops.cast(y_true, tf.float32)
        dist = keras.ops.sqrt(keras.ops.sum(keras.ops.square(y_pred), axis=-1))
        
        # Contrastive loss formula
        loss = keras.ops.mean(0.5*(1-y_true) * keras.ops.square(dist) + y_true *
                               0.5*keras.ops.square(keras.ops.maximum(margin - dist, 0)))
        return loss
    return loss

def create_LSTM_constrastive(input_shape):
    base_model = create_LSTM_network(input_shape)

    input_1 = keras.layers.Input(shape=input_shape)
    input_2 = keras.layers.Input(shape=input_shape)

    embedding_1 = base_model(input_1)
    embedding_2 = base_model(input_2)

    # Calculate the absolute difference between the two outputs
    distance = layers.Lambda(lambda tensors: keras.ops.abs(tensors[0] - tensors[1]))([embedding_1, embedding_2])

    # Define the model
    model = keras.Model(inputs=[input_1, input_2], outputs=distance)

    return model

def triplet_loss(margin=1.0):

    #y_true is not used in this case, y_pred is a collection of embedding trios
    def loss(y_true, y_pred):
        # Split the predictions into anchor, positive, and negative embeddings
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)

        pos_distance = keras.ops.sum(keras.ops.square(anchor - positive), axis=1)
        neg_distance = keras.ops.sum(keras.ops.square(anchor - negative), axis=1)
        loss = keras.ops.maximum(pos_distance - neg_distance + margin, 0.0)

        return keras.ops.mean(loss)
    
    return loss

def create_LSTM_triplet(input_shape):
    # Create the embedding model
    base_model = create_LSTM_network(input_shape)

    # Define three inputs: anchor, positive, negative
    anchor_input = keras.layers.Input(shape=input_shape, name='anchor')
    positive_input = keras.layers.Input(shape=input_shape, name='positive')
    negative_input = keras.layers.Input(shape=input_shape, name='negative')

    # Get embeddings for each input
    anchor_embedding = base_model(anchor_input)
    positive_embedding = base_model(positive_input)
    negative_embedding = base_model(negative_input)

    # Merge embeddings into a single tensor(need to check which dimension)
    #merged_output = keras.layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=1)
    #merged_output = keras.layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=1)
    merged_output = keras.layers.Concatenate(axis=1)([anchor_embedding, positive_embedding, negative_embedding])

    # Create the model
    model = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output)
    return model


# tens = np.array([[1,2,3],[4,5,6],[7,8,9]])
# a,b,c = tf.split(tens,3,axis=0)
# print(a)
# a,b,c = tf.split(tens,3,axis=1)
# print(a)

seq_len = 70
num_features = 5
num_users_softmax = 10000
# num_users_softmax = 1000
input_shape = (seq_len,num_features)


#random data for syntax purposes, replace with real data later
x_train = np.random.random((num_users_softmax*15, seq_len,num_features))
y_train = tf.keras.utils.to_categorical(np.random.randint(low=1,high=num_users_softmax,size=num_users_softmax*15),
                                         num_classes=num_users_softmax)
x_test = np.random.random((num_users_softmax*15, seq_len,num_features))
y_test = np.random.randint(low=1,high=num_users_softmax,size=num_users_softmax*15)

padded_x_train = keras.utils.pad_sequences(x_train, padding='post', dtype='float32', value=0)

# kek = [[[1,2,3],[2,3,4],[5,5,6]],[[1,2,3],[2,3,4],[5,5,6]],[[1,2,3],[2,3,4]]]
# padded_kek = keras.utils.pad_sequences(kek, padding='post',maxlen=5, dtype='float32', value=0)
# print(padded_kek)

model = create_LSTM_softmax_network(input_shape=input_shape,num_users=num_users_softmax)
#print(model.summary())
optimizer = keras.optimizers.Adam(learning_rate=0.05,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=200,steps_per_epoch=150,batch_size=512)
model.save("softmax_typenet_model.keras")

import datetime
import pytz
eastern = pytz.timezone('US/Eastern')
fmt = '%Y-%m-%d %H:%M:%S %Z%z'
loc_dt = datetime.datetime.now(eastern)
print("model saved at: ",loc_dt.strftime(fmt))

# paul comment start
# model_1 = create_LSTM_triplet(input_shape=input_shape)
# print(model_1.summary())

# model_1.save("saved_model.keras")

# model_2 = keras.models.load_model("saved_model.keras")
# print(model_2.summary())

# lst = np.array([[1,2],[3,4]])
# lst_2 = np.array([[5,6],[7,8]])
# print(np.concatenate((lst,lst_2)))
# paul comment end

# padded_x_test = keras.utils.pad_sequences(x_train, padding='post', dtype='float32', value=-1)
# predictions = model.evaluate(x_test)
# predicted_labels = np.argmax(predictions,axis=-1)

class BalancedPairGenerator(keras.utils.Sequence):
    def __init__(self, x1, x2, y, batch_size):

        self.x1_same = [x1[i] for i in range(len(y)) if y[i] == 0]
        self.x2_same = [x2[i] for i in range(len(y)) if y[i] == 0]
        self.x1_diff = [x1[i] for i in range(len(y)) if y[i] == 1]
        self.x2_diff = [x2[i] for i in range(len(y)) if y[i] == 1]
        self.batch_size = batch_size
        self.indices_same = np.arange(len(self.x1_same))
        self.indices_diff = np.arange(len(self.x1_diff))

    def __len__(self):
        # Number of batches per epoch(might need to change)
        return int(np.floor(min(len(self.x1_same), len(self.x1_diff))/self.batch_size))

    def __getitem__(self, index):
        same_batch_indexes = self.indices_same[index*self.batch_size//2:(index + 1)*self.batch_size//2]
        diff_batch_indexes = self.indices_diff[index*self.batch_size//2:(index + 1)*self.batch_size//2]
        batch_x1_same = np.array([self.x1_same[i] for i in same_batch_indexes])
        batch_x2_same = np.array([self.x2_same[i] for i in same_batch_indexes])
        batch_x1_diff = np.array([self.x1_diff[i] for i in diff_batch_indexes])
        batch_x2_diff = np.array([self.x2_diff[i] for i in diff_batch_indexes])
        batch_x1 = np.concatenate([batch_x1_same, batch_x1_diff])
        batch_x2 = np.concatenate([batch_x2_same, batch_x2_diff])
        batch_y = np.concatenate([np.zeros(len(batch_x1_same)), np.ones(len(batch_x1_diff))])

        # Shuffle the batch to ensure randomness
        indices = np.arange(len(batch_x1))
        np.random.shuffle(indices)
        batch_x1 = batch_x1[indices]
        batch_x2 = batch_x2[indices]
        batch_y = batch_y[indices]

        return [batch_x1, batch_x2], batch_y

    def on_epoch_end(self):
        # Shuffle the indexes at the end of each epoch to ensure randomness in the next epoch
        #(should we shuffle negative pairs less frequently?)
        np.random.shuffle(self.indices_same)
        np.random.shuffle(self.indices_diff)


#pick a batch size(or check in typenet paper)
x1 = x_train  
x2 = x_train
x3 = x_train
y = np.random.randint(low=0,high=1,size=num_users_softmax*15)  # Labels (0 for same user, 1 for different users)

#contrastive loss

# batch_size = 32
# pair_generator = BalancedPairGenerator(x1, x2, y, batch_size)
# model = create_LSTM_constrastive(input_shape=input_shape)
# model.compile(optimizer='adam', loss=contrastive_loss(margin=1.0))
# model.fit(pair_generator, epochs=30)

# paul comment start
# #triplet loss
# model = create_LSTM_triplet(input_shape=input_shape)
# model.compile(optimizer='adam',loss=triplet_loss(margin=1.0))
# print(model.summary())
# #model.fit(x=np.array([x1,x2,x3]),y=None,epochs=200,steps_per_epoch=150,batch_size=512)
# model.fit(x=[x1,x2,x3],y=y,epochs=200,steps_per_epoch=150,batch_size=512)
# paul comment end