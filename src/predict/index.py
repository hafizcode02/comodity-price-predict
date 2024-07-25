# Define the API endpoint for making predictions, predictions for the next x days, 
# where x is the number of steps in the future we want to predict.
# The endpoint expects a POST request with a JSON payload containing the last 30 observations.
# The endpoint returns a JSON response containing the predicted prices for the next x days.
# The endpoint uses the pre-trained model and scaler to make predictions.
# The correct data input is only used on the first day prediction, on the second day and next the first day prediction and next (output) will be used as new data to be used for prediction.
# example : 0-29 days data input, predict for day 30. day 30 prediction will be used as new data input for day 31 prediction, and so on.

from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import joblib
import pandas as pd
import os

# Create a new Blueprint for the predict API
predict = Blueprint('predict', __name__)

# Define the path to the dataset
dataset_path = 'storage/dataset.xlsx'

# Load the pre-trained model and scaler
model_bawang_merah = load_model('storage/model_bawang_merah.h5')
scaler_bawang_merah = joblib.load('storage/scaler_bawang_merah.pkl')

model_cabai_merah_besar = load_model('storage/model_bawang_merah.h5')
scaler_cabai_merah_besar = joblib.load('storage/scaler_bawang_merah.pkl')

model_cabai_merah_keriting = load_model('storage/model_bawang_merah.h5')
scaler_cabai_merah_keriting = joblib.load('storage/scaler_bawang_merah.pkl')

model_cabai_rawit_hijau = load_model('storage/model_bawang_merah.h5')
scaler_cabai_rawit_hijau = joblib.load('storage/scaler_bawang_merah.pkl')

model_cabai_rawit_merah = load_model('storage/model_bawang_merah.h5')
scaler_cabai_rawit_merah = joblib.load('storage/scaler_bawang_merah.pkl')

# Base Route
@predict.route('/api')
def index():
    return "API - Comodity Price Predict"

# Validate Request JSON
def validate_request_data(required_fields):
    data = request.get_json(force=True)
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]

    if missing_fields:
        return jsonify({"error": f"Missing or null fields: {', '.join(missing_fields)}"}), 400

    return data

# Predict API
@predict.route('/api/predict-bawang-merah', methods=['POST'])
def predict_bawang_merah():
    # Required Fields
    required_fields = ['days_prediction', 'use_raw_data', 'last_observations']
    validation_result = validate_request_data(required_fields)
    
    if isinstance(validation_result, tuple):  # If it's a tuple, it's an error response
        return validation_result
    
    # Fetch the data from the validation result
    data = validation_result
    days_prediction = data['days_prediction']
    use_raw_data = data['use_raw_data']
    last_observations = data['last_observations']

    # Ensure we have the right number of observations
    window_size = 30
    
    # Check if use the raw data
    if use_raw_data == True:
        if len(last_observations) != window_size:
            return jsonify({'error': 'Provide exactly {} last observations'.format(window_size)}), 400
        
        # Scale the input data from last observations
        scaled_data = scaler_bawang_merah.transform(np.array(last_observations).reshape(-1, 1))
    else:
        # Load the dataset
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found, upload the .xlsx file first'}), 404
        
        df = pd.read_excel(dataset_path)
        df_sorted = df.sort_values(by='date')
        df_sorted_tail = df_sorted.tail(window_size)
        scaled_data = scaler_bawang_merah.transform(df_sorted_tail['bawang_merah'].values.reshape(-1, 1))

    # Prepare the input data for prediction
    future_data = scaled_data[-window_size:, :] # 30 last observations

    # Predict the next days
    future_steps = days_prediction
    future_predictions = []
    for _ in range(future_steps):
        # reshape the input data (30, 1) to (1, 30, 1) 
        # which is the input shape of the model (batch_size, timesteps, features)
        future_input = future_data.reshape((1, window_size, 1))
        # make a prediction
        future_pred = model_bawang_merah.predict(future_input)
        # append the prediction to the list
        future_predictions.append(future_pred[0, 0])
        # update the input data for the next prediction, using the current prediction
        future_data = np.vstack((future_data[1:], future_pred))

    # Inverse transform the predictions to the original scale
    future_predictions = scaler_bawang_merah.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

    # Return the predictions as a JSON response
    return jsonify({'predictions': future_predictions})

@predict.route('/api/predict-cabai-merah-besar', methods=['POST'])
def predict_cabai_merah_besar():
    data = request.get_json(force=True)
    last_observations = data['last_observations']
    
    # Ensure we have the right number of observations
    window_size = 30
    if len(last_observations) != window_size:
        return jsonify({'error': 'Provide exactly {} last observations'.format(window_size)}), 400

    # Scale the input data
    scaled_data = scaler_cabai_merah_besar.transform(np.array(last_observations).reshape(-1, 1))

    # Prepare the input data for prediction
    future_data = scaled_data[-window_size:, :]

    # Predict the next days
    future_steps = 1
    future_predictions = []
    for _ in range(future_steps):
        # reshape the input data (30, 1) to (1, 30, 1) 
        # which is the input shape of the model (batch_size, timesteps, features)
        future_input = future_data.reshape((1, window_size, 1))
        # make a prediction
        future_pred = model_cabai_merah_besar.predict(future_input)
        # append the prediction to the list
        future_predictions.append(future_pred[0, 0])
        # update the input data for the next prediction, using the current prediction
        future_data = np.vstack((future_data[1:], future_pred))

    # Inverse transform the predictions to the original scale
    future_predictions = scaler_cabai_merah_besar.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

    # Return the predictions as a JSON response
    return jsonify({'predictions': future_predictions})

@predict.route('/api/predict-cabai-merah-keriting', methods=['POST'])
def predict_cabai_merah_keriting():
    data = request.get_json(force=True)
    last_observations = data['last_observations']
    
    # Ensure we have the right number of observations
    window_size = 30
    if len(last_observations) != window_size:
        return jsonify({'error': 'Provide exactly {} last observations'.format(window_size)}), 400

    # Scale the input data
    scaled_data = scaler_cabai_merah_keriting.transform(np.array(last_observations).reshape(-1, 1))

    # Prepare the input data for prediction
    future_data = scaled_data[-window_size:, :]

    # Predict the next days
    future_steps = 1
    future_predictions = []
    for _ in range(future_steps):
        # reshape the input data (30, 1) to (1, 30, 1) 
        # which is the input shape of the model (batch_size, timesteps, features)
        future_input = future_data.reshape((1, window_size, 1))
        # make a prediction
        future_pred = model_cabai_merah_keriting.predict(future_input)
        # append the prediction to the list
        future_predictions.append(future_pred[0, 0])
        # update the input data for the next prediction, using the current prediction
        future_data = np.vstack((future_data[1:], future_pred))

    # Inverse transform the predictions to the original scale
    future_predictions = scaler_cabai_merah_keriting.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

    # Return the predictions as a JSON response
    return jsonify({'predictions': future_predictions})

@predict.route('/api/predict-cabai-rawit-hijau', methods=['POST'])
def predict_cabai_rawit_hijau():
    data = request.get_json(force=True)
    last_observations = data['last_observations']
    
    # Ensure we have the right number of observations
    window_size = 30
    if len(last_observations) != window_size:
        return jsonify({'error': 'Provide exactly {} last observations'.format(window_size)}), 400

    # Scale the input data
    scaled_data = scaler_cabai_rawit_hijau.transform(np.array(last_observations).reshape(-1, 1))

    # Prepare the input data for prediction
    future_data = scaled_data[-window_size:, :]

    # Predict the next days
    future_steps = 1
    future_predictions = []
    for _ in range(future_steps):
        # reshape the input data (30, 1) to (1, 30, 1) 
        # which is the input shape of the model (batch_size, timesteps, features)
        future_input = future_data.reshape((1, window_size, 1))
        # make a prediction
        future_pred = model_cabai_rawit_hijau.predict(future_input)
        # append the prediction to the list
        future_predictions.append(future_pred[0, 0])
        # update the input data for the next prediction, using the current prediction
        future_data = np.vstack((future_data[1:], future_pred))

    # Inverse transform the predictions to the original scale
    future_predictions = scaler_cabai_rawit_hijau.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

    # Return the predictions as a JSON response
    return jsonify({'predictions': future_predictions})

@predict.route('/api/predict-cabai-rawit-merah', methods=['POST'])
def predict_cabai_rawit_merah():
    data = request.get_json(force=True)
    last_observations = data['last_observations']
    
    # Ensure we have the right number of observations
    window_size = 30
    if len(last_observations) != window_size:
        return jsonify({'error': 'Provide exactly {} last observations'.format(window_size)}), 400

    # Scale the input data
    scaled_data = scaler_cabai_rawit_merah.transform(np.array(last_observations).reshape(-1, 1))

    # Prepare the input data for prediction
    future_data = scaled_data[-window_size:, :]

    # Predict the next days
    future_steps = 1
    future_predictions = []
    for _ in range(future_steps):
        # reshape the input data (30, 1) to (1, 30, 1)
        # which is the input shape of the model (batch_size, timesteps, features)
        future_input = future_data.reshape((1, window_size, 1))
        # make a prediction
        future_pred = model_cabai_rawit_merah.predict(future_input)
        # append the prediction to the list
        future_predictions.append(future_pred[0, 0])
        # update the input data for the next prediction, using the current prediction
        future_data = np.vstack((future_data[1:], future_pred))

    # Inverse transform the predictions to the original scale
    future_predictions = scaler_cabai_rawit_merah.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

    # Return the predictions as a JSON response
    return jsonify({'predictions': future_predictions})
