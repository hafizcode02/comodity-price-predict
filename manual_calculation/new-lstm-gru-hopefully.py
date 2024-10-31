import numpy as np
import joblib

# # Load the MinMaxScaler object from the pickle file using joblib
# try:
#     scaler = joblib.load("E:\Braincore\skripsi\comodity-price-predict\manual_calculation\scaler.pkl")
#     print("Loaded MinMaxScaler object")
# except FileNotFoundError:
#     print("Error: 'scal.pkl' file not found. Please ensure the file exists in the specified directory.")
#     exit()  # Stop the execution if the file is not found
# except Exception as e:
#     print(f"An error occurred while loading the pickle file: {e}")
#     exit()

min_price_data = 19000
max_price_data = 70000
    
price_data = np.array([45000,45000,45000,45000,45000,45000,45000,45000,45000,45000,65000,65000,65000,65000,65000,65000,65000,60000,60000,60000,60000,60000,60000,60000,60000,57500,55000,55000,53333,51667])

# Data input
timesteps = 30
features = 1

# Input tensor (shape: (30, 1))
# input_tensor = np.random.rand(timesteps, features)
# input_tensor = scaler.transform(price_data.reshape(-1, 1))
input_tensor = (price_data - min_price_data) / (max_price_data - min_price_data)
input_tensor = input_tensor.reshape((1, 30, 1))
input_tensor = np.round(input_tensor, 6)  # Round off to 6 decimal places

print("Price tensor data:", input_tensor)

# --- LSTM Layer ---
# LSTM: units = 2, input_dim = 1
units_lstm = 2
input_dim = 1

# Kernel and recurrent kernel weights (randomly initialized)
# Kernel shape: (input_dim, 4 * units_lstm)
kernel_lstm = np.array([
    [-0.5059566, 1.0306768, 0.8995212, 0.11026055, 0.20074809, 0.94807816, 0.28200635, 0.7189556 ]
])
# Recurrent kernel shape: (units_lstm, 4 * units_lstm)
recurrent_kernel_lstm = np.array([
   [ 0.15625614, -0.20001842, 0.34109056, 0.5909459, 0.34335765, 0.61546695, -0.09291892, -0.9629568 ], 
   [ 0.6239177, -0.05001342, 0.6716121, -0.20007247, -0.5731505, -0.37635723, 0.2940814, 0.3249863 ]
])
# Bias shape: (4 * units_lstm)
bias_lstm = np.array([
    0.05210945, 0.28922087, 1.0046924, 1.0428534, -0.08918256, -0.04497227, 0.10883114, 0.2909315
])

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
    h_lstm, c_lstm = lstm_step(input_tensor[0,t], h_lstm, c_lstm)
    output_lstm.append(h_lstm)

output_lstm = np.array(output_lstm)  # Shape (30, 2)

print("Output of the LSTM layer (shape: (30, 2)):\n", output_lstm)

# --- GRU Layer ---
# GRU: units = 2, input_dim = 2 (output from LSTM)
units_gru = 2

# Kernel and recurrent kernel weights (randomly initialized)
# Kernel shape: (input_dim, 3 * units_gru)
kernel_gru = np.array([
    [ 2.0781596,  0.35970873, -0.66276604, -0.74533767, 0.0452804, -0.47344992],
    [-1.1140864, -0.3651697, -0.44320884, -0.2677313, 0.44454804, 0.42388165]
])
# Recurrent kernel shape: (units_gru, 3 * units_gru)
recurrent_kernel_gru = np.array([
    [-0.8664823, -0.4084303, 0.7153386, 0.28564006, 0.94721794, 0.14692006], 
    [-1.722285, -1.4838053, 0.9102488, 0.275525, -0.4610603, 0.07878926]
])

# Bias shape: (2, 6), splitting into two parts
bias_gru_reset_update = np.array([
    [-1.188044, -1.0855407,  0.0815194,  0.09651377, -0.01091218, 0.03348876]
])
bias_gru_candidate = np.array([
    [-1.188044, -1.0855407,  0.0815194,  0.09651377, -0.04127811, 0.01440805]
])

# # GRU cell computations (with customized bias handling)
# def gru_step(x, h_prev):
#     z = np.dot(x, kernel_gru) + np.dot(h_prev, recurrent_kernel_gru)
    
#     # Split into reset, update, and candidate gates
#     r, z, h_bar = np.split(z, 3, axis=-1)
    
#     # Apply the bias components for reset/update and candidate gates
#     r += bias_gru_reset_update[:, :units_gru].reshape(-1)  # Reshape bias to match shape of r
#     z += bias_gru_reset_update[:, units_gru:2*units_gru].reshape(-1)  # Reshape bias to match shape of z
#     h_bar += bias_gru_candidate[:, 2*units_gru:].reshape(-1)  # Reshape bias to match shape of h_bar

#     r = 1 / (1 + np.exp(-r))  # Sigmoid
#     z = 1 / (1 + np.exp(-z))  # Sigmoid
#     h_bar = np.tanh(np.dot(x, kernel_gru[:, 2*units_gru:]) + r * np.dot(h_prev, recurrent_kernel_gru[:, 2*units_gru:]))
#     h = z * h_prev + (1 - z) * h_bar
#     return h

# Revised GRU cell computations
def gru_step(x, h_prev):
    z = np.dot(x, kernel_gru) + np.dot(h_prev, recurrent_kernel_gru)
    r, z, h_bar = np.split(z, 3, axis=-1)
    r += bias_gru_reset_update[:, :units_gru].reshape(-1)
    r = 1 / (1 + np.exp(-r))
    z += bias_gru_reset_update[:, units_gru:2*units_gru].reshape(-1)
    z = 1 / (1 + np.exp(-z))
    h_bar = np.tanh(
        np.dot(x, kernel_gru[:, 2*units_gru:]) +
        r * np.dot(h_prev, recurrent_kernel_gru[:, 2*units_gru:]) +
        bias_gru_candidate[:, 2*units_gru:].reshape(-1)
    )
    h = z * h_prev + (1 - z) * h_bar
    return h

# Initialize GRU hidden states
h_gru = np.zeros((units_gru,))

# Process the input tensor through GRU (return_sequences=False)
for t in range(timesteps):
    h_gru = gru_step(output_lstm[t], h_gru)

# Final GRU output (shape: (2,))
output_gru = h_gru

print("Output of the GRU layer (shape: (2,)):", output_gru)

# --- Dense Layer ---
# Dense: units = 1, input_dim = 2 (output from GRU)
output_dim = 1

# Dense layer weights (randomly initialized)
# Kernel shape: (input_dim, output_dim)
kernel_dense = np.array([
    [0.5349816], [1.6744679]
])
# Bias shape: (output_dim)
bias_dense = np.array([
    0.03799994
])
# Dense output computation
output_dense = np.dot(output_gru, kernel_dense) + bias_dense  # Shape (1,)

print("Output of the model:", output_dense)

denormalized_output = output_dense * (max_price_data - min_price_data) + min_price_data

print("Final output of the model:", denormalized_output)
