import numpy as np

# Data input
timesteps = 30
features = 1

# Input tensor (shape: (30, 1))
input_tensor = np.random.rand(timesteps, features)

# --- LSTM Layer ---
# LSTM: units = 2, input_dim = 1
units_lstm = 2
input_dim = 1

# Kernel and recurrent kernel weights (randomly initialized)
# Kernel shape: (input_dim, 4 * units_lstm)
kernel_lstm = np.random.rand(input_dim, 4 * units_lstm)  # (1, 8)
# Recurrent kernel shape: (units_lstm, 4 * units_lstm)
recurrent_kernel_lstm = np.random.rand(units_lstm, 4 * units_lstm)  # (2, 8)
# Bias shape: (4 * units_lstm)
bias_lstm = np.random.rand(4 * units_lstm)  # (8,)

# LSTM cell computations (simplified)
def lstm_step(x, h_prev, c_prev):
    z = np.dot(x, kernel_lstm) + np.dot(h_prev, recurrent_kernel_lstm) + bias_lstm
    i, f, c_bar, o = np.split(z, 4, axis=-1)  # Split into 4 gates
    i = 1 / (1 + np.exp(-i))  # Sigmoid
    f = 1 / (1 + np.exp(-f))  # Sigmoid
    o = 1 / (1 + np.exp(-o))  # Sigmoid
    c_bar = np.tanh(c_bar)    # Tanh
    c = f * c_prev + i * c_bar
    h = o * np.tanh(c)
    return h, c

# Initialize LSTM hidden states
h_lstm = np.zeros((units_lstm,))
c_lstm = np.zeros((units_lstm,))

# Process the input tensor through LSTM (return_sequences=True)
output_lstm = []
for t in range(timesteps):
    h_lstm, c_lstm = lstm_step(input_tensor[t], h_lstm, c_lstm)
    output_lstm.append(h_lstm)

output_lstm = np.array(output_lstm)  # Shape (30, 2)

# --- GRU Layer ---
# GRU: units = 2, input_dim = 2 (output from LSTM)
units_gru = 2

# Kernel and recurrent kernel weights (randomly initialized)
# Kernel shape: (input_dim, 3 * units_gru)
kernel_gru = np.random.rand(units_lstm, 3 * units_gru)  # (2, 6)
# Recurrent kernel shape: (units_gru, 3 * units_gru)
recurrent_kernel_gru = np.random.rand(units_gru, 3 * units_gru)  # (2, 6)

# Bias shape: (2, 6), splitting into two parts
bias_gru_reset_update = np.random.rand(1, 3 * units_gru)  # (1, 6)
bias_gru_candidate = np.random.rand(1, 3 * units_gru)     # (1, 6)

# GRU cell computations (with customized bias handling)
def gru_step(x, h_prev):
    z = np.dot(x, kernel_gru) + np.dot(h_prev, recurrent_kernel_gru)
    
    # Split into reset, update, and candidate gates
    r, z, h_bar = np.split(z, 3, axis=-1)
    
    # Apply the bias components for reset/update and candidate gates
    r += bias_gru_reset_update[:, :units_gru].reshape(-1)  # Reshape bias to match shape of r
    z += bias_gru_reset_update[:, units_gru:2*units_gru].reshape(-1)  # Reshape bias to match shape of z
    h_bar += bias_gru_candidate[:, 2*units_gru:].reshape(-1)  # Reshape bias to match shape of h_bar

    r = 1 / (1 + np.exp(-r))  # Sigmoid
    z = 1 / (1 + np.exp(-z))  # Sigmoid
    h_bar = np.tanh(np.dot(x, kernel_gru[:, 2*units_gru:]) + r * np.dot(h_prev, recurrent_kernel_gru[:, 2*units_gru:]))
    h = z * h_prev + (1 - z) * h_bar
    return h

# Initialize GRU hidden states
h_gru = np.zeros((units_gru,))

# Process the input tensor through GRU (return_sequences=False)
for t in range(timesteps):
    h_gru = gru_step(output_lstm[t], h_gru)

# Final GRU output (shape: (2,))
output_gru = h_gru

# --- Dense Layer ---
# Dense: units = 1, input_dim = 2 (output from GRU)
output_dim = 1

# Dense layer weights (randomly initialized)
# Kernel shape: (input_dim, output_dim)
kernel_dense = np.random.rand(units_gru, output_dim)  # (2, 1)
# Bias shape: (output_dim)
bias_dense = np.random.rand(output_dim)  # (1,)

# Dense output computation
output_dense = np.dot(output_gru, kernel_dense) + bias_dense  # Shape (1,)

print("Final output of the model:", output_dense)
