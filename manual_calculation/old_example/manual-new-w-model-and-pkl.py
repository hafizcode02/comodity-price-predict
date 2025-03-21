import numpy as np
import joblib

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
    z0, z1, z2 = np.split(z, 3, axis=-1)  # Split into update gate, reset gate, and candidate

    update_gate = sigmoid(z0)  # Update gate
    reset_gate = sigmoid(z1)  # Reset gate
    h_tilde = tanh(z2)  # Candidate hidden state

    h_next = (1 - update_gate) * h_prev + update_gate * h_tilde

    return h_next

# Load the MinMaxScaler object from the pickle file using joblib
try:
    scaler = joblib.load('E:/Braincore/skripsi/comodity-price-predict/manual_calculation/scalar.pkl')
    print("Loaded MinMaxScaler object")
except FileNotFoundError:
    print("Error: 'scal.pkl' file not found. Please ensure the file exists in the specified directory.")
    exit()  # Stop the execution if the file is not found
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")
    exit()

# Example input: Price for 30 days
price_data = np.array([41000, 45000, 45000, 45000, 45000, 45000, 45000, 45000, 45000, 45000, 45000, 45000, 65000, 65000, 65000, 65000, 65000, 65000, 65000, 60000, 60000, 60000, 60000, 60000, 60000, 60000, 60000, 60000, 55000, 55000])

# Normalize the data using the scaler
normalized_data = scaler.transform(price_data.reshape(-1, 1))
normalized_data = normalized_data.reshape((1, 30, 1))  # Reshape to (batch_size, time_steps, features)

# Initialize weights and biases (convert lists to NumPy arrays)
lstm_units = 4
gru_units = 4
output_units = 1

# Convert lists to NumPy arrays
lstm_kernel = np.array([
    [
        0.051292452961206436,
        0.791134238243103,
        0.5069447755813599,
        -0.01498005073517561,
        0.34781137108802795,
        -0.4243960678577423,
        0.6344618201255798,
        0.5929720401763916,
        0.43543052673339844,
        -0.3988112807273865,
        -0.28907185792922974,
        -0.5989620089530945,
        0.07997361570596695,
        0.13263384997844696,
        1.2702420949935913,
        0.30399230122566223
    ]
])
lstm_recurrent_kernel = np.array([
    [
        -0.03213103488087654,
        -0.15782266855239868,
        0.47003987431526184,
        0.5161075592041016,
        0.1741826981306076,
        0.021259427070617676,
        0.4244462251663208,
        -0.52400141954422,
        0.07416681945323944,
        0.003138095373287797,
        -0.334199458360672,
        0.25283578038215637,
        -0.09156136959791183,
        0.5552884936332703,
        -0.01012980006635189,
        0.21423138678073883
    ],
    [
        -0.22469498217105865,
        -0.11160020530223846,
        -0.16624987125396729,
        0.19798198342323303,
        0.20750103890895844,
        -0.4050058424472809,
        0.05676260590553284,
        -0.151491180062294,
        0.21049454808235168,
        -0.267277330160141,
        -0.1786869466304779,
        -0.6157670021057129,
        -0.42868566513061523,
        -0.7414934635162354,
        -0.4087581932544708,
        0.121955007314682
    ],
    [
        -0.14041990041732788,
        -0.6117061972618103,
        -0.46769019961357117,
        -0.8834439516067505,
        -0.19413350522518158,
        -0.32708099484443665,
        -0.23776577413082123,
        -0.20682229101657867,
        0.16937915980815887,
        -0.38413944840431213,
        0.05462295562028885,
        -0.22509288787841797,
        -0.18649445474147797,
        -0.7228811979293823,
        -0.5101325511932373,
        -0.728498637676239
    ],
    [
        -0.16484421491622925,
        -0.4157927930355072,
        -0.46943873167037964,
        -0.5728983283042908,
        0.027591990306973457,
        -0.01729586347937584,
        0.04087232053279877,
        -0.43981581926345825,
        -0.06287451833486557,
        0.33646225929260254,
        0.7328222393989563,
        -0.4192201793193817,
        -0.11424817144870758,
        -0.2766689956188202,
        -0.5225928425788879,
        -0.10178330540657043
    ]
])
lstm_bias = np.array([
    -0.012561949901282787,
    0.42065128684043884,
    0.39679667353630066,
    0.42822253704071045,
    0.9435327649116516,
    0.9911412596702576,
    1.2442818880081177,
    0.9787622690200806,
    -0.01666211150586605,
    -0.00659980671480298,
    -0.1759311854839325,
    0.03607456386089325,
    0.019764183089137077,
    0.42313313484191895,
    0.08208989351987839,
    0.37329867482185364
])


# GRU weights
gru_kernel = np.array([
    [
        -0.755038321018219,
        -0.5140049457550049,
        -0.6804197430610657,
        -1.1402360200881958,
        -0.33693379163742065,
        -0.2769099175930023,
        -0.2083428055047989,
        0.25643807649612427,
        0.1272079348564148,
        -0.03715407848358154,
        0.2925870716571808,
        -0.6149566769599915
    ],
    [
        -0.2208591103553772,
        0.5008842349052429,
        0.025904200971126556,
        -0.2700279951095581,
        0.8575302362442017,
        1.0126936435699463,
        0.8253057599067688,
        -0.19434437155723572,
        0.7504489421844482,
        -0.6101908683776855,
        -0.8179451823234558,
        0.6749907732009888
    ],
    [
        0.8242130875587463,
        0.8187986016273499,
        1.4492326974868774,
        0.7302184700965881,
        0.022148024290800095,
        0.5899961590766907,
        0.007901654578745365,
        0.7143163681030273,
        0.21552787721157074,
        -0.2840719223022461,
        -0.3322125971317291,
        0.683531641960144
    ],
    [
        0.42886802554130554,
        0.6721928119659424,
        1.304073691368103,
        0.4958379864692688,
        0.2318451851606369,
        0.9677062630653381,
        0.6133027672767639,
        0.10135455429553986,
        -0.14177657663822174,
        -0.5227148532867432,
        -0.8543000817298889,
        0.2187126725912094
    ]
])

gru_recurrent_kernel = np.array([
    [
        0.2560883164405823,
        0.6046854257583618,
        1.415814995765686,
        0.8700149655342102,
        -0.07514576613903046,
        0.07049943506717682,
        0.10780568420886993,
        -0.12741787731647491,
        -0.3828785717487335,
        0.164909228682518,
        0.521464467048645,
        0.2054327130317688
    ],
    [
        -0.6208978891372681,
        -1.1383349895477295,
        -1.0579259395599365,
        -0.30087414383888245,
        -0.13666431605815887,
        -0.014116212725639343,
        0.03889673948287964,
        -0.5212523341178894,
        0.16580554842948914,
        -0.5654686093330383,
        -0.43032488226890564,
        -0.08938821405172348
    ],
    [
        -0.476396381855011,
        -0.7209393382072449,
        -0.2011544555425644,
        -0.38134586811065674,
        -0.7116858959197998,
        -0.13001073896884918,
        -0.1280927062034607,
        0.46151041984558105,
        0.3851332664489746,
        0.22069354355335236,
        -0.293317973613739,
        0.11170703172683716
    ],
    [
        0.9816722869873047,
        1.2593762874603271,
        1.6845139265060425,
        1.4936858415603638,
        -0.23892098665237427,
        0.1343630999326706,
        -0.298994779586792,
        -0.03573162853717804,
        -0.2352188229560852,
        0.04260358214378357,
        0.15082068741321564,
        -0.16868956387043
    ]
])

gru_bias = np.array([
    -0.51313716173172,
    -1.101780652999878,
    -1.3163409233093262,
    -1.007932424545288,
    -0.10984373092651367,
    0.01909417286515236,
    0.08096685260534286,
    -0.1313510239124298,
    -0.03614501655101776,
    0.02223019115626812,
    -0.11811506748199463,
    -0.00009578063327353448
])

# Dense layer weights
dense_kernel = np.array([
    [
        -0.3986716568470001
    ],
    [
        1.1943769454956055
    ],
    [
        0.3501167297363281
    ],
    [
        -0.3544594347476959
    ]
])
dense_bias = np.array([
    0.027977393940091133
])

# LSTM forward pass for all time steps
h_lstm = np.zeros((1, 30, lstm_units))
c_lstm = np.zeros((1, lstm_units))

# LSTM forward pass for all time steps
for t in range(30):
    h_lstm[:, t, :], c_lstm = lstm_forward(
        normalized_data[:, t, :], h_lstm[:, t-1, :] if t > 0 else np.zeros((1, lstm_units)), c_lstm,
        lstm_kernel, lstm_recurrent_kernel, lstm_bias
    )

# GRU forward pass
h_gru = np.zeros((1, gru_units))

for t in range(30):
    h_gru = gru_forward(
        h_lstm[:, t, :], h_gru,
        gru_kernel, gru_recurrent_kernel, gru_bias
    )

# Dense layer
output = np.dot(h_gru, dense_kernel) + dense_bias

# Denormalize the output using the scaler
denormalized_output = scaler.inverse_transform(output)

print("Denormalized Model Output:", denormalized_output)
