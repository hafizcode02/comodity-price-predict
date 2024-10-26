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
    [
        -0.5059565901756287,
        1.0306768417358398,
        0.8995211720466614,
        0.11026055365800858,
        0.20074808597564697,
        0.9480781555175781,
        0.2820063531398773,
        0.7189555764198303
    ]
])
# Recurrent kernel shape: (units_lstm, 4 * units_lstm)
recurrent_kernel_lstm = np.array([
   [
        0.15625613927841187,
        -0.20001842081546783,
        0.3410905599594116,
        0.5909458994865417,
        0.343357652425766,
        0.6154669523239136,
        -0.09291891753673553,
        -0.9629567861557007
    ],
    [
        0.6239176988601685,
        -0.05001341551542282,
        0.6716120839118958,
        -0.20007246732711792,
        -0.5731505155563354,
        -0.37635722756385803,
        0.2940813899040222,
        0.3249863088130951
    ]
])
# Bias shape: (4 * units_lstm)
bias_lstm = np.array([
    0.05210945010185242,
    0.2892208695411682,
    1.0046924352645874,
    1.0428533554077148,
    -0.08918255567550659,
    -0.04497227072715759,
    0.10883113741874695,
    0.29093149304389954
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
   [
        2.0781595706939697,
        0.3597087264060974,
        -0.6627660393714905,
        -0.7453376650810242,
        0.04528040066361427,
        -0.4734499156475067
    ],
    [
        -1.114086389541626,
        -0.3651697039604187,
        -0.44320884346961975,
        -0.26773130893707275,
        0.4445480406284332,
        0.4238816499710083
    ]
])
# Recurrent kernel shape: (units_gru, 3 * units_gru)
recurrent_kernel_gru = np.array([
   [
        -0.8664823174476624,
        -0.4084303081035614,
        0.7153385877609253,
        0.28564006090164185,
        0.9472179412841797,
        0.14692005515098572
    ],
    [
        -1.7222850322723389,
        -1.4838052988052368,
        0.9102488160133362,
        0.2755250036716461,
        -0.4610602855682373,
        0.07878925651311874
    ]
])

# Bias shape: (2, 6), splitting into two parts
bias_gru_reset_update = np.array([
    [
        -1.1880439519882202,
        -1.0855406522750854,
        0.08151939511299133,
        0.0965137705206871,
        -0.010912179946899414,
        0.03348875790834427
    ],
])
bias_gru_candidate = np.array([
    [
        -1.1880439519882202,
        -1.0855406522750854,
        0.08151939511299133,
        0.0965137705206871,
        -0.041278108954429626,
        0.014408046379685402
    ]
])

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

# # Revised GRU cell computations
# def gru_step(x, h_prev):
#     z = np.dot(x, kernel_gru) + np.dot(h_prev, recurrent_kernel_gru)
#     r, z, h_bar = np.split(z, 3, axis=-1)
#     r += bias_gru_reset_update[:, :units_gru].reshape(-1)
#     r = 1 / (1 + np.exp(-r))
#     z += bias_gru_reset_update[:, units_gru:2*units_gru].reshape(-1)
#     z = 1 / (1 + np.exp(-z))
#     h_bar = np.tanh(
#         np.dot(x, kernel_gru[:, 2*units_gru:]) +
#         r * np.dot(h_prev, recurrent_kernel_gru[:, 2*units_gru:]) +
#         bias_gru_candidate[:, 2*units_gru:].reshape(-1)
#     )
#     h = z * h_prev + (1 - z) * h_bar
#     return h

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
     [
        0.5349816083908081
    ],
    [
        1.674467921257019
    ]
])
# Bias shape: (output_dim)
bias_dense = np.array([
    0.03799993917346001
])
# Dense output computation
output_dense = np.dot(output_gru, kernel_dense) + bias_dense  # Shape (1,)

print("Output of the model:", output_dense)

denormalized_output = output_dense * (max_price_data - min_price_data) + min_price_data

print("Final output of the model:", denormalized_output)
