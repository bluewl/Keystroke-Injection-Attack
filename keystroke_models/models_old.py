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
import datetime
import pytz

#global variable that holds all user galleries
galleries_dict= {}

#try removing recurrent dropout for speedups (CuDNN acceleration)
def create_LSTM_network(input_shape):
    input = layers.Input(shape=input_shape)
    #input = layers.Input(shape=(input_shape,5))
    masked = layers.Masking(mask_value=0.0)(input)
    normalized_masked = layers.BatchNormalization()(input)
    first_LSTM = layers.LSTM(units=128,activation="tanh",
    recurrent_activation="sigmoid",dropout=0.5,recurrent_dropout=0.2,return_sequences=True)(normalized_masked)
    # first_LSTM = layers.LSTM(units=128,activation="tanh",
    # recurrent_activation="sigmoid",dropout=0.5,return_sequences=True)(normalized_masked)
    normalized_first_LSTM = layers.BatchNormalization()(first_LSTM)
    second_LSTM = layers.LSTM(units=128,activation="tanh",
    recurrent_activation="sigmoid",recurrent_dropout=0.2)(normalized_first_LSTM)
    # second_LSTM = layers.LSTM(units=128,activation="tanh",
    # recurrent_activation="sigmoid")(normalized_first_LSTM)
    return keras.Model(input, second_LSTM)


#if we pad, we'd have a list of S sequences, each represented by an M*5 matrix
#what needs to be padded is the M dimension S*M*S
#pad with -1 just to be safe
#keras.layers.Masking(mask_value=-1.0, **kwargs)
#padded_sequences = pad_sequences(sequences, padding='post', dtype='float32', value=0)

#remember to remove recurrent dropout if it's of no use!
#maybe try all tanh instead of sigmoid?
def create_LSTM_softmax_network(input_shape,num_users):
    #input = layers.Input(shape=input_shape)
    #input = layers.Input(shape=(input_shape,5))
    input = layers.Input(shape=input_shape)
    masked = layers.Masking(mask_value=0.0)(input)
    normalized_masked = layers.BatchNormalization()(masked)
    first_LSTM = layers.LSTM(units=128,activation="tanh",
    recurrent_activation="sigmoid",dropout=0.5,recurrent_dropout=0.2,return_sequences=True)(normalized_masked)
    # first_LSTM = layers.LSTM(units=128,activation="tanh",
    # recurrent_activation="sigmoid",dropout=0.5,return_sequences=True)(normalized_masked)
    normalized_first_LSTM = layers.BatchNormalization()(first_LSTM)
    second_LSTM = layers.LSTM(units=128,activation="tanh",
    recurrent_activation="sigmoid",recurrent_dropout=0.2)(normalized_first_LSTM)
    # second_LSTM = layers.LSTM(units=128,activation="tanh",
    # recurrent_activation="sigmoid")(normalized_first_LSTM)
    #logits = layers.Dense(units=num_users,activation="relu")(second_LSTM)
    outputs = layers.Dense(units=num_users,activation="softmax")(second_LSTM)
    return keras.Model(input, outputs)

# def create_LSTM_softmax_network(input_shape, num_users):
#     input = layers.Input(shape=input_shape)
#     x = layers.Masking(mask_value=0.0)(input)
#     x = layers.LayerNormalization()(x)

#     x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
#     x = layers.LayerNormalization()(x)
#     x = layers.LSTM(128)(x)

#     output = layers.Dense(num_users, activation='softmax',
#                           kernel_regularizer=keras.regularizers.l2(0.001))(x)
    
#     return keras.Model(input, output)


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

    return model , base_model

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
    return model , base_model

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

        #return [batch_x1, batch_x2], batch_y
        return (batch_x1, batch_x2), batch_y

    def on_epoch_end(self):
        # Shuffle the indexes at the end of each epoch to ensure randomness in the next epoch
        #(should we shuffle negative pairs less frequently?)
        np.random.shuffle(self.indices_same)
        np.random.shuffle(self.indices_diff)

#embedding distance between two sequences(todo: possibly extend to multiple sequences)
def embedding_distance(model,seq_1,seq_2):
    embed_1 = model(np.array([seq_1])) if seq_1.ndim==2 else model(seq_1)
    #embed_1 = model(np.array([seq_1]))
    embed_2 = model(np.array([seq_2])) if seq_1.ndim==2 else model(seq_2)
    #embed_2 = model(seq_2)
    # Compute the Euclidean distance between embeddings
    dist = keras.ops.sqrt(keras.ops.sum(keras.ops.square(embed_1-embed_2), axis=-1))
    # print(dist)
    # print("reeeturninnngg")
    result = dist.numpy()[0] if len(dist.numpy())==1 else dist.numpy()
    return result 

#for single user
def is_user(test_embedding,gallery_embedding,threshold=1):
    gallery_embedding =  np.array([gallery_embedding]) if gallery_embedding.ndim == 1 else gallery_embedding
    test_embedding =  np.array([test_embedding]) if test_embedding.ndim == 1 else test_embedding
    dist = gallery_embedding - test_embedding
    dist = np.mean(np.sum(dist ** 2, axis = -1))
    #print(dist)
    if (dist<=threshold):
        return True
    return False

#for one user
#construct a dictionary of embeddings beforehand
def compute_FPR_FNR(test_embeddings,alleged_users,gallery_embeddings_dict,threshold=1):
    positive_samples = 5
    negative_samples = len(test_embeddings) -5
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0 
    for i in range(5):
        # print(alleged_users[i])
        if is_user(test_embeddings[i],gallery_embeddings_dict[alleged_users[i]],threshold):
            true_positives += 1
        else:
            false_negatives += 1
    for j in range(5,len(test_embeddings)):
        if is_user(test_embeddings[j],gallery_embeddings_dict[alleged_users[j]],threshold):
            # print(alleged_users[j])
            false_positives += 1
        else:
            true_negatives += 1
    #todo: calculate false positive rate
    false_positive_rate = false_positives/(false_positives+true_negatives)
    false_negative_rate = false_negatives/(false_negatives+true_positives)
    return (false_positive_rate,false_negative_rate)

#takes in dictionaries mapping users to their test_samples and alleged users
def compute_avg_FPR_FNR(test_embeddings_dict,alleged_users_dict,gallery_embeddings_dict,threshold=1):
    avg_FPR = 0
    avg_FNR = 0
    num_users = len(test_embeddings_dict.keys())
    for user in test_embeddings_dict.keys():
        temp_FPR,temp_FNR = compute_FPR_FNR(test_embeddings_dict[user],alleged_users_dict[user],gallery_embeddings_dict,threshold)
        avg_FPR += temp_FPR
        avg_FNR += temp_FNR

    avg_FPR /= num_users
    avg_FNR /= num_users
    return (avg_FPR,avg_FNR,abs(avg_FPR-avg_FNR))

def compute_EER(test_embeddings_dict,alleged_users_dict,gallery_embeddings_dict):
    #thresholds = np.linspace(0,2,2000) #should this be the right range?
    thresholds = np.linspace(0,2,10) #budget version
    lst = []
    min_diff = -5
    best_FPR = -5
    best_FNR = -5
    for threshold in thresholds:
        temp_FPR,temp_FNR,diff = compute_avg_FPR_FNR(test_embeddings_dict,alleged_users_dict,gallery_embeddings_dict,threshold)
        if min_diff == -5 or diff < min_diff:
            min_diff = diff
            best_FPR = temp_FPR
            best_FNR = temp_FNR

    return (best_FPR+best_FNR)/2 #can also return best_FNR, or best FPR    


seq_len = 70
num_features = 5
#num_users_softmax = 10000
num_users_softmax = 1000
input_shape = (seq_len,num_features)


#random data for syntax purposes, replace with real data later
# x_train = np.random.random((num_users_softmax*15, seq_len,num_features))
# y_train = tf.keras.utils.to_categorical(np.random.randint(low=0,high=num_users_softmax,size=num_users_softmax*15),
#                                          num_classes=num_users_softmax)
# x_test = np.random.random((num_users_softmax*15, seq_len,num_features))
# y_test = np.random.randint(low=0,high=num_users_softmax,size=num_users_softmax*15)
txt_file = open("MyFile","a")
# with np.printoptions(threshold=np.inf):
#     print(y_test)
# padded_x_train = keras.utils.pad_sequences(x_train, padding='post', dtype='float32', value=0)

print("hiiiiii")
saved_model= keras.models.load_model("softmax_typenet_model_dummy.keras")
#saved_model= keras.models.load_model("softmax_typenet_model_1.keras")
#saved_model = keras.models.load_model("softmax_typenet_model_1.keras")
 
# model = create_LSTM_softmax_network(input_shape=input_shape,num_users=num_users_softmax)
# #print(model.summary())
# optimizer = keras.optimizers.Adam(learning_rate=0.05,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
# model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
# print(y_train.shape)
# model.fit(x_train,y_train,epochs=200,batch_size=512)
# saved_model = model

# all_embeddings = []
# #length = len(x_test)
# for i in range(15):
#     print(saved_model(x_test[num_users_softmax*i:num_users_softmax*(i+1)]).numpy().shape)
#     temp = list(saved_model(x_test[num_users_softmax*i:num_users_softmax*(i+1)]).numpy())
#     all_embeddings.extend(temp)
# # all_embeddings = saved_model(x_test)
# # all_embeddings = all_embeddings.numpy()
# all_embeddings = np.array(all_embeddings)
# print(all_embeddings.shape)
# user_embeddings_dict = {}
# test_embeddings_dict = {}
# gallery_embeddings_dict = {}
# alleged_users_dict = {}

# test_lst = []
# user_lst = np.array([i for i in range(num_users_softmax)])

# print(x_test)
#print(all_embeddings)
# for i in range(len(all_embeddings)-1):
#     if(list(all_embeddings[i])!=list(all_embeddings[i+1])):
#         print("FALSEEEEEEE")
# for i in range(num_users_softmax):
#     print(i)
#     #indices = tf.convert_to_tensor(np.nonzero(y_test == i)[0])
#     indices = np.nonzero(y_test == i)[0]
#     user_embeddings_dict[i] = all_embeddings[indices]
#     gallery_embeddings_dict[i] = all_embeddings[indices][:10]
#     #test_embeddings_dict[i] = list(all_embeddings[indices][10:])
#     #we sore indices in test_embeddings_dict so we don't have to create new numpy arrays
#     test_embeddings_dict[i] = list(indices)
#     alleged_users_dict[i] = [i] * 5
#     test_lst.append(np.random.choice(indices))

# test_lst = np.array(test_lst)

# for i in range(num_users_softmax):
#     print(i)
#     indices = np.nonzero(y_test == i)[0]
#     #temp_user = [j for j in range(num_users_softmax) if j!=i]
#     temp_user = [j for j in range(num_users_softmax) if j!=i][:100]
#     temp_test = list(test_lst[:i])
#     print(len(temp_test))
#     #print(temp_test.shape)
#     alleged_users_dict[i].extend(temp_user)
#     test_embeddings_dict[i].extend(temp_test)
#     temp_test = list(test_lst[i+1:])
#     test_embeddings_dict[i].extend(temp_test)
#     # for j in range(num_users_softmax):
#     #     if(i != j):
#     #         alleged_users_dict[j].append(i)
#     #         random_index = np.random.choice(indices) #move outside loop if too inefficient
#     #         test_embeddings_dict[j].append(all_embeddings[random_index])

# for i in range(num_users_softmax):
#     print(i)
#     #print(test_embeddings_dict[i])
#     #all_embeddings[np.array(test_embeddings_dict[i])]
#     # np.array(alleged_users_dict[i])
#     # np.array(test_embeddings_dict[i])
#     alleged_users_dict[i] = np.array(alleged_users_dict[i])
#     test_embeddings_dict[i] = all_embeddings[test_embeddings_dict[i][:105]]
#     #print(test_embeddings_dict[i].shape)

# for i in tqdm(range(10000*10000)):
#     i+1
# #print(compute_FPR_FNR(test_embeddings,alleged_users,gallery_embeddings_dict,threshold=1))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=10000000))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=256))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=128))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=64))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=32))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=16))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=8))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=4))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=2))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=1))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=0.5))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=0.25))
# print(compute_FPR_FNR(test_embeddings_dict[0],alleged_users_dict[0],gallery_embeddings_dict,threshold=0.125))
print('sajfnjidsnfkjndsjfnjksdnfkn')

#saved_model= keras.models.load_model("base_saved_model.keras")
#print(embedding_distance(saved_model,np.array(x_train[0]),np.array(x_train[2])))
#print("reeeeeeeeeeeeeeeeeeeeeeeee")


# #tf.profiler.experimental.start('logdir')
# model = create_LSTM_softmax_network(input_shape=input_shape,num_users=num_users_softmax)
# #print(model.summary())
# optimizer = keras.optimizers.Adam(learning_rate=0.05,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
# model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
# #model.fit(x_train,y_train,epochs=200,steps_per_epoch=150,batch_size=512)
# print(x_train.shape)
# print(y_train.shape)
# model.fit(x_train,y_train,epochs=200,batch_size=512)
# #model.fit(x_train,y_train,epochs=1,steps_per_epoch=150,batch_size=512)
# #model.save("softmax_typenet_model.keras")
# model.save("softmax_typenet_model_dummy.keras")
# #tf.profiler.experimental.stop()

eastern = pytz.timezone('US/Eastern')
fmt = '%Y-%m-%d %H:%M:%S %Z%z'
loc_dt = datetime.datetime.now(eastern)
print("model saved at: ",loc_dt.strftime(fmt))

#testing verification functions
# print("testinggggg")
# for i in range(10):
#     galleries_dict[i]= x_test[10*i:10*i+10]
# #test_model = keras.models.load_model("softmax_typenet_model_dummy.keras")
# test_model = create_LSTM_softmax_network(input_shape=input_shape,num_users=num_users_softmax)
# actual_users_list = sorted(galleries_dict.keys())
# actual_users = [[i for j in range(len(galleries_dict[i]))] for i in actual_users_list]
# new_actual_users = []
# for lst in actual_users:
#     new_actual_users.extend(lst)
# print(len(new_actual_users))
# actual_users = new_actual_users

# print(actual_users)
# alleged_users = np.random.choice(actual_users_list,size = len(actual_users),replace=True)
# print(actual_users_list)
# test_lst = actual_users[0:5]
# test_lst.extend(actual_users[10:15])
# print(compute_FPR_FNR(test_model,galleries_dict[0],test_lst,galleries_dict,threshold=1))
#print(compute_FPR_FNR(test_model,galleries_dict[0],alleged_users[0:10],galleries_dict,threshold=1))

# test_dict = {}
# for user in actual_users:
#     test_dict[user] = galleries_dict[user]
# alleged_users_dict = {}
# for user in actual_users:
#     alleged_users_dict[user] = test_lst
# print(compute_avg_FPR_FNR(test_model,test_dict,alleged_users_dict,galleries_dict,threshold=0.5))
# print(compute_EER(test_model,test_dict,alleged_users_dict,galleries_dict))
# print("testing doneeee")


# paul comment start
# model_1 = create_LSTM_triplet(input_shape=input_shape)
# print(model_1.summary())

# model_1.save("saved_model.keras")

# model_2 = keras.models.load_model("saved_model.keras")
# print(model_2.summary())

# paul comment end

# padded_x_test = keras.utils.pad_sequences(x_train, padding='post', dtype='float32', value=-1)
# predictions = model.evaluate(x_test)
# predicted_labels = np.argmax(predictions,axis=-1)


#pick a batch size(or check in typenet paper)
# x1 = x_test  
# x2 = x_test
# x3 = x_test
# y = np.random.randint(low=0,high=2,size=num_users_softmax*15)  # Labels (0 for same user, 1 for different users)

#contrastive loss

#batch_size = 32
# print(y)
# #batch_size = 32
# batch_size = 512
# pair_generator = BalancedPairGenerator(x1, x2, y, batch_size)
# model,base_model = create_LSTM_constrastive(input_shape=input_shape)
# model.compile(optimizer='adam', loss=contrastive_loss(margin=1.0))
# #model.fit(pair_generator, epochs=30)
# model.fit(pair_generator, epochs=200)

# paul comment start
# #triplet loss
# model,base_model = create_LSTM_triplet(input_shape=input_shape)
# model.compile(optimizer='adam',loss=triplet_loss(margin=1.0))
# print(model.summary())
# #model.fit(x=np.array([x1,x2,x3]),y=None,epochs=200,steps_per_epoch=150,batch_size=512)
# #model.fit(x=[x1,x2,x3],y=y,epochs=200,steps_per_epoch=150,batch_size=512)
# model.fit(x=[x1,x2,x3],y=y,epochs=200,batch_size=512)
# paul comment end