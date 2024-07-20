import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# LSTM forward pass
def lstm_forward(x, h_prev, c_prev, kernel, recurrent_kernel, bias):
    z = np.dot(x, kernel) + np.dot(h_prev, recurrent_kernel) + bias
    z0, z1, z2, z3 = np.split(z, 4, axis=-1)
    
    i = sigmoid(z0)  # input gate
    f = sigmoid(z1)  # forget gate
    c = np.tanh(z2)  # cell gate
    o = sigmoid(z3)  # output gate
    
    c_next = f * c_prev + i * c
    h_next = o * np.tanh(c_next)
    
    return h_next, c_next

# GRU forward pass
def gru_forward(x, h_prev, kernel, recurrent_kernel, bias):
    z = np.dot(x, kernel) + np.dot(h_prev, recurrent_kernel) + bias
    z0, z1, z2 = np.split(z, 3, axis=-1)
    
    z = sigmoid(z0)  # update gate
    r = sigmoid(z1)  # reset gate
    h_tilde = np.tanh(np.dot(x, kernel[:, gru_units:2*gru_units]) + r * np.dot(h_prev, recurrent_kernel[:, gru_units:2*gru_units]) + bias[gru_units:2*gru_units])
    
    h_next = (1 - z) * h_prev + z * h_tilde
    
    return h_next

# Example input: Price for 30 days
price_data = np.array([
    100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
    111, 110, 112, 114, 113, 115, 117, 116, 118, 120,
    119, 121, 123, 122, 124, 126, 125, 127, 129, 128
])

# Normalize the data
normalized_data = (price_data - np.min(price_data)) / (np.max(price_data) - np.min(price_data))
normalized_data = normalized_data.reshape((1, 30, 1))  # Reshape to (batch_size, time_steps, features)

# Initialize weights and biases
lstm_units = 64
gru_units = 64
dense_units = 32
output_units = 1

# LSTM weights
lstm_kernel = np.random.randn(1, 4 * lstm_units)  # (input_features, 4 * units)
lstm_recurrent_kernel = np.random.randn(lstm_units, 4 * lstm_units)  # (units, 4 * units)
lstm_bias = np.random.randn(4 * lstm_units)  # (4 * units)

# GRU weights
gru_kernel = np.random.randn(lstm_units, 3 * gru_units)  # (input_features, 3 * units)
gru_recurrent_kernel = np.random.randn(gru_units, 3 * gru_units)  # (units, 3 * units)
gru_bias = np.random.randn(3 * gru_units)  # (3 * units)

# Dense layer weights
dense_kernel = np.random.randn(gru_units, dense_units)  # (input_features, units)
dense_bias = np.random.randn(dense_units)  # (units)

# Output layer weights
output_kernel = np.random.randn(dense_units, output_units)  # (input_features, units)
output_bias = np.random.randn(output_units)  # (units)

# LSTM forward pass for all time steps
h_lstm = np.zeros((1, 30, lstm_units))
c_lstm = np.zeros((1, lstm_units))

for t in range(30):
    h_lstm[:, t, :], c_lstm = lstm_forward(
        normalized_data[:, t, :], h_lstm[:, t-1, :] if t > 0 else np.zeros((1, lstm_units)), c_lstm,
        lstm_kernel, lstm_recurrent_kernel, lstm_bias
    )

# Dropout after LSTM
h_lstm *= np.random.binomial([np.ones_like(h_lstm)], 1 - 0.2)[0] * (1.0 / (1 - 0.2))

# GRU forward pass
h_gru = np.zeros((1, gru_units))

for t in range(30):
    h_gru = gru_forward(
        h_lstm[:, t, :], h_gru,
        gru_kernel, gru_recurrent_kernel, gru_bias
    )

# Dropout after GRU
h_gru *= np.random.binomial([np.ones_like(h_gru)], 1 - 0.2)[0] * (1.0 / (1 - 0.2))

# Dense layer
dense_output = np.dot(h_gru, dense_kernel) + dense_bias
dense_output = np.maximum(0, dense_output)  # ReLU activation

# Output layer
output = np.dot(dense_output, output_kernel) + output_bias

print("Model Output:", output)
