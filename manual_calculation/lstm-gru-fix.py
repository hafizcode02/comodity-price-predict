import numpy as np
import joblib
import tensorflow as tf

min_price_data = 19000
max_price_data = 70000
    
price_data = np.array([45000,45000,45000,45000,45000,45000,45000,45000,45000,45000,65000,65000,65000,65000,65000,65000,65000,60000,60000,60000,60000,60000,60000,60000,60000,57500,55000,55000,53333,51667])

# Data input
timesteps = 30
features = 1

input_tensor = (price_data - min_price_data) / (max_price_data - min_price_data)
input_tensor = input_tensor.reshape((1, 30, 1))
input_tensor = np.round(input_tensor, 6)  # Round off to 6 decimal places

# print("Price tensor data:", input_tensor)

# --- LSTM Layer ---
# LSTM: units = 2, input_dim = 1
units_lstm = 2
input_dim = 1

# Kernel shape: (input_dim, 4 * units_lstm)
kernel_lstm = np.array([
    [
        -0.07712285965681076,
        0.006820258218795061,
        -0.6268875002861023,
        0.26155000925064087,
        0.405324250459671,
        -0.9832772016525269,
        -0.2981899082660675,
        1.2878139019012451
    ]
])
# Recurrent kernel shape: (units_lstm, 4 * units_lstm)
recurrent_kernel_lstm = np.array([
    [
        -0.3449821174144745,
        0.2094593048095703,
        0.5368510484695435,
        0.8081979155540466,
        0.2004215121269226,
        -0.07392501085996628,
        0.45246556401252747,
        1.4950858354568481
    ],
    [
        -0.6041944026947021,
        0.4682267904281616,
        -0.06551726162433624,
        -0.02143552340567112,
        0.37362945079803467,
        -0.279207706451416,
        -0.3175261616706848,
        -0.4961036145687103
    ]
])
# Bias shape: (4 * units_lstm)
bias_lstm = np.array([
    0.30134350061416626,
    0.34250497817993164,
    0.9535882472991943,
    0.9420087337493896,
    -0.09631551057100296,
    0.014600558206439018,
    0.312594473361969,
    0.2818651795387268
])

iterate = 0

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
    
    # if(iterate == 0 or iterate == 1):
    #     print("Iteration: ", iterate)
    #     print("=====================================")
        
    #     print("")
    #     print("------------------ Ht-1 & Ct-1 ------------------")
    #     print("h_prev:", h_prev)
    #     print("c_bar:", c_prev)
    #     print("-------------------------------------------------")
    #     print("")
        
    #     print("------------------ Breakdown Value ------------------")
    #     print("x:", x)
    #     print("kernel_lstm:", kernel_lstm)
    #     print("x.kernel_lstm:",np.dot(x, kernel_lstm))
    #     print("")
        
    #     print("h_prev:", h_prev)
    #     print("recurrent_kernel_lstm:", recurrent_kernel_lstm)
    #     print("h_prev.reccurent_kernel_lstm:",np.dot(h_prev, recurrent_kernel_lstm))
    #     print("")
        
    #     print("bias_lstm: ",bias_lstm)
    #     print("")
        
    #     print("z:", z)
        
    #     print("------------------------------------------------------")  
        
             
    #     print("i:", i)
    #     print("f:", f)
    #     print("c_bar:", c_bar)
    #     print("c:", c)
    #     print("o:", o)
    #     print("h:", h)
    #     print("=====================================")
    #     print("")

    return h, c
    

# Initialize LSTM hidden states
h_lstm = np.zeros((units_lstm,))
c_lstm = np.zeros((units_lstm,))

# Process the input tensor through LSTM (return_sequences=True)
output_lstm = []
for t in range(timesteps):
    h_lstm, c_lstm = lstm_step(input_tensor[0,t], h_lstm, c_lstm)
    output_lstm.append(h_lstm)
    iterate += 1

output_lstm = np.array(output_lstm)  # Shape (30, 2)

print("Output of the LSTM layer (shape: (30, 2)):\n", output_lstm)

# --- GRU Layer ---
# GRU: units = 2, input_dim = 2 (output from LSTM)
units_gru = 2

# Kernel and recurrent kernel weights (randomly initialized)
# Kernel shape: (input_dim, 3 * units_gru)
kernel_gru = np.array([
    [
        0.6967200040817261,
        0.593103289604187,
        0.08725280314683914,
        1.0769436359405518,
        0.7281477451324463,
        0.44972744584083557
    ],
    [
        1.8398795127868652,
        0.6037645936012268,
        -0.7010221481323242,
        0.01506162341684103,
        -0.21882161498069763,
        -0.23427139222621918
    ]
])

# Recurrent kernel shape: (units_gru, 3 * units_gru)
recurrent_kernel_gru = np.array([
    [
        -1.009032130241394,
        -0.7375137805938721,
        0.15157319605350494,
        0.009581056423485279,
        0.797579824924469,
        -0.18758957087993622
    ],
    [
        -0.9052043557167053,
        -1.5690317153930664,
        0.3640286922454834,
        -0.24158819019794464,
        0.1588675081729889,
        0.8135177493095398
    ]
])

# Bias GRU : 
bias_gru = np.array([
    -1.3423137664794922,
    -1.4769455194473267,
    0.05379534512758255,
    0.16837242245674133,
    0.04781540855765343,
    0.06481347233057022
])

def gru_step(x, h_prev):
    # Step 1: Compute the update gate (z)
    Wz = kernel_gru[:, :units_gru]  # Weights for input to update gate
    Uz = recurrent_kernel_gru[:, :units_gru]  # Recurrent weights for update gate
    bz = bias_gru[:units_gru]  # Bias for update gate
    z = np.dot(x, Wz) + np.dot(h_prev, Uz) + bz  # Linear combination
    z = 1 / (1 + np.exp(-z))  # Apply sigmoid activation
    # print("\n--- Update Gate (z) ---")
    # print("x:", x)
    # print("Wz:", Wz)
    # print("x @ Wz:", np.dot(x, Wz))
    # print("h_prev:", h_prev)
    # print("Uz:", Uz)
    # print("h_prev @ Uz:", np.dot(h_prev, Uz))
    # print("bz:", bz)
    # print("Linear combination (z):", np.dot(x, Wz) + np.dot(h_prev, Uz) + bz)
    # print("z (after sigmoid):", z)

    # Step 2: Compute the reset gate (r)
    Wr = kernel_gru[:, units_gru:2*units_gru]  # Weights for input to reset gate
    Ur = recurrent_kernel_gru[:, units_gru:2*units_gru]  # Recurrent weights for reset gate
    br = bias_gru[units_gru:2*units_gru]  # Bias for reset gate
    r = np.dot(x, Wr) + np.dot(h_prev, Ur) + br  # Linear combination
    r = 1 / (1 + np.exp(-r))  # Apply sigmoid activation
    # print("\n--- Reset Gate (r) ---")
    # print("x:", x)
    # print("Wr:", Wr)
    # print("x @ Wr:", np.dot(x, Wr))
    # print("h_prev:", h_prev)
    # print("Ur:", Ur)
    # print("h_prev @ Ur:", np.dot(h_prev, Ur))
    # print("br:", br)
    # print("Linear combination (r):", np.dot(x, Wr) + np.dot(h_prev, Ur) + br)
    # print("r (after sigmoid):", r)

    # Step 3: Compute the candidate hidden state (h_tilde)
    Wh = kernel_gru[:, 2*units_gru:]  # Weights for input to candidate hidden state
    Uh = recurrent_kernel_gru[:, 2*units_gru:]  # Recurrent weights for candidate hidden state
    bh = bias_gru[2*units_gru:]  # Bias for candidate hidden state
    h_tilde = np.dot(x, Wh) + np.dot(r * h_prev, Uh) + bh  # Linear combination
    h_tilde = np.tanh(h_tilde)  # Apply tanh activation
    # print("\n--- Candidate Hidden State (h_tilde) ---")
    # print("x:", x)
    # print("Wh:", Wh)
    # print("x @ Wh:", np.dot(x, Wh))
    # print("r * h_prev:", r * h_prev)
    # print("Uh:", Uh)
    # print("(r * h_prev) @ Uh:", np.dot(r * h_prev, Uh))
    # print("bh:", bh)
    # print("Linear combination (h_tilde):", np.dot(x, Wh) + np.dot(r * h_prev, Uh) + bh)
    # print("h_tilde (after tanh):", h_tilde)

    # Step 4: Compute the new hidden state (h)
    # h = (1 - z) * h_prev + z * h_tilde
    h = z * h_prev + (1 - z) * h_tilde
    # print("\n--- New Hidden State (h) ---")
    # print("1 - z:", 1 - z)
    # print("(1 - z) * h_prev:", (1 - z) * h_prev)
    # print("z * h_tilde:", z * h_tilde)
    # print("h (new hidden state):", h)

    return h


# Initialize GRU hidden states
h_gru = np.zeros((units_gru,))

print("h_gru: ", h_gru)

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
        1.0679084062576294
    ],
    [
        1.1822028160095215
    ]
])
# Bias shape: (output_dim)
bias_dense = np.array([
    0.06151740998029709
])
# Dense output computation
output_dense = np.dot(output_gru, kernel_dense) + bias_dense  # Shape (1,)

print("Output of the model:", output_dense)

denormalized_output = output_dense * (max_price_data - min_price_data) + min_price_data

print("Final output of the model:", denormalized_output)
