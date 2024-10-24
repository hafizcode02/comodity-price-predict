import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh function
def tanh(x):
    return np.tanh(x)

# LSTM forward pass
def lstm_forward(x, h_prev, c_prev, kernel, recurrent_kernel, bias):
    z = np.dot(x, kernel) + np.dot(h_prev, recurrent_kernel) + bias
    z0, z1, z2, z3 = np.split(z, 4, axis=-1)
    
    i = sigmoid(z0)  # input gate
    f = sigmoid(z1)  # forget gate
    c_tilde = tanh(z2)  # cell gate
    o = sigmoid(z3)  # output gate
    
    c_next = f * c_prev + i * c_tilde
    h_next = o * tanh(c_next)
    
    return h_next, c_next

# GRU forward pass
def gru_forward(x, h_prev, kernel, recurrent_kernel, bias):
    z = np.dot(x, kernel) + np.dot(h_prev, recurrent_kernel) + bias
    z0, z1, h_tilde_kernel = np.split(z, [gru_units, 2 * gru_units], axis=-1)
    
    update_gate = sigmoid(z0)  # update gate
    reset_gate = sigmoid(z1)  # reset gate
    h_tilde = tanh(np.dot(x, kernel[:, -gru_units:]) + reset_gate * np.dot(h_prev, recurrent_kernel[:, -gru_units:]) + bias[-gru_units:])
    
    h_next = (1 - update_gate) * h_prev + update_gate * h_tilde
    
    return h_next

# Example input: Price for 30 days
price_data = np.array([
    100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
    111, 110, 112, 114, 113, 115, 117, 116, 118, 120,
    119, 121, 123, 122, 124, 126, 125, 127, 129, 128
])

# Normalize the data
min_value = np.min(price_data)
max_value = np.max(price_data)
normalized_data = (price_data - min_value) / (max_value - min_value)
normalized_data = normalized_data.reshape((1, 30, 1))  # Reshape to (batch_size, time_steps, features)

# Initialize weights and biases
lstm_units = 2
gru_units = 2
output_units = 1

# LSTM weights
lstm_kernel = np.random.randn(1, 4 * lstm_units)  # (input_features, 4 * units)
print("LSTM Kernel : ", lstm_kernel)
lstm_recurrent_kernel = np.random.randn(lstm_units, 4 * lstm_units)  # (units, 4 * units)
print("LSTM Reccurent Kernel : ", lstm_recurrent_kernel)
lstm_bias = np.random.randn(4 * lstm_units)  # (4 * units)
print("LSTM Bias : ", lstm_bias)
print("==============================================")

# GRU weights
gru_kernel = np.random.randn(lstm_units, 3 * gru_units)  # (lstm_units, 3 * gru_units)
print("GRU Kernel : ", gru_kernel)
gru_recurrent_kernel = np.random.randn(gru_units, 3 * gru_units)  # (gru_units, 3 * gru_units)
print("GRU Reccurent Kernel : ", gru_recurrent_kernel)
gru_bias = np.random.randn(3 * gru_units)  # (3 * gru_units)
print("GRU Bias : ", gru_bias)
print("==============================================")

# Dense layer weights
dense_kernel = np.random.randn(gru_units, output_units)  # (gru_units, output_units)
print("Dense Kernel : ", dense_kernel)
dense_bias = np.random.randn(output_units)  # (output_units)
print("Dense Bias : ", dense_bias)
print("==============================================")

# LSTM forward pass for all time steps
h_lstm = np.zeros((1, 30, lstm_units))
print("H LSTM : ", h_lstm)
c_lstm = np.zeros((1, lstm_units))
print("C LSTM : ", c_lstm)

for t in range(30):
    h_lstm[:, t, :], c_lstm = lstm_forward(
        normalized_data[:, t, :], h_lstm[:, t-1, :] if t > 0 else np.zeros((1, lstm_units)), c_lstm,
        lstm_kernel, lstm_recurrent_kernel, lstm_bias
    )

print("H LSTM : ", h_lstm)
print("")

# GRU forward pass
h_gru = np.zeros((1, gru_units))

for t in range(30):
    h_gru = gru_forward(
        h_lstm[:, t, :], h_gru,
        gru_kernel, gru_recurrent_kernel, gru_bias
    )

# Dense layer
output = np.dot(h_gru, dense_kernel) + dense_bias

# Denormalize the output
denormalized_output = (output * (max_value - min_value)) + min_value

print("Denormalized Model Output:", denormalized_output)
