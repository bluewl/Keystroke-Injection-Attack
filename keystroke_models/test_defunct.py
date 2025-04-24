import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Define a custom contrastive loss function
def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        # Calculate the Euclidean distance between the embeddings
        squared_pred = tf.square(y_pred[0] - y_pred[1])
        distance = tf.sqrt(tf.maximum(tf.reduce_sum(squared_pred, axis=1), 1e-6))

        # Contrastive loss formula
        positive_loss = y_true * tf.square(distance)
        negative_loss = (1 - y_true) * tf.maximum(margin - distance, 0.0)
        return tf.reduce_mean(positive_loss + negative_loss)
    
    return loss

# Define the model architecture with two LSTM layers
class LSTMContrastiveModel(Model):
    def __init__(self, lstm_units=64, input_dim=5):
        super(LSTMContrastiveModel, self).__init__()
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True)
        self.lstm2 = layers.LSTM(lstm_units, return_sequences=False)
        self.dense = layers.Dense(lstm_units, activation='relu')
        self.embedding = layers.Dense(lstm_units)

    def call(self, inputs):
        input1, input2 = inputs
        x1 = self.lstm1(input1)
        x1 = self.lstm2(x1)
        x1 = self.dense(x1)
        embedding1 = self.embedding(x1)

        x2 = self.lstm1(input2)
        x2 = self.lstm2(x2)
        x2 = self.dense(x2)
        embedding2 = self.embedding(x2)

        return [embedding1, embedding2]

# Generate synthetic data for illustration
def generate_data(num_samples=1000, seq_length=10, input_dim=5):
    # Random binary labels: 1 for positive pairs, 0 for negative pairs
    X1 = np.random.randn(num_samples, seq_length, input_dim)
    X2 = np.random.randn(num_samples, seq_length, input_dim)
    y = np.random.randint(2, size=num_samples)  # Binary labels
    
    return X1, X2, y

# Set up the dataset with positive and negative pairs
def create_balanced_batch(X1, X2, y, batch_size=32):
    # Separate positive and negative samples
    positive_pairs = [(X1[i], X2[i], y[i]) for i in range(len(y)) if y[i] == 1]
    negative_pairs = [(X1[i], X2[i], y[i]) for i in range(len(y)) if y[i] == 0]
    
    # Shuffle and balance the batch
    np.random.shuffle(positive_pairs)
    np.random.shuffle(negative_pairs)

    positive_batch = positive_pairs[:batch_size // 2]
    negative_batch = negative_pairs[:batch_size // 2]
    
    batch_X1 = np.array([x[0] for x in positive_batch + negative_batch])
    batch_X2 = np.array([x[1] for x in positive_batch + negative_batch])
    batch_y = np.array([x[2] for x in positive_batch + negative_batch])

    return batch_X1, batch_X2, batch_y

# Initialize and compile the model
input_dim = 5
seq_length = 10
lstm_units = 64
model = LSTMContrastiveModel(lstm_units=lstm_units, input_dim=input_dim)
model.compile(optimizer='adam', loss=contrastive_loss(margin=1.0))

# Generate data
X1, X2, y = generate_data(num_samples=1000, seq_length=seq_length, input_dim=input_dim)

# Train the model
batch_size = 32
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    # Create balanced batches
    for i in range(0, len(X1), batch_size):
        batch_X1, batch_X2, batch_y = create_balanced_batch(X1[i:i+batch_size], X2[i:i+batch_size], y[i:i+batch_size], batch_size)
        # Train on the current batch
        loss = model.train_on_batch([batch_X1, batch_X2], batch_y)
        print(f"Batch loss: {loss:.4f}")

