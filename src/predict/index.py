from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib

predict = Blueprint('predict', __name__)

# Load the pre-trained model and scaler
model_bawang_merah = load_model('storage/model_bawang_merah.h5')
scaler = joblib.load('storage/scaler.pkl')

@predict.route('/api')
def index():
    return "API - Comodity Price Predict"

@predict.route('/api/predict-bawang-merah', methods=['POST'])
def predict_bawang_merah():
    data = request.get_json(force=True)
    last_observations = data['last_observations']

    # Ensure we have the right number of observations
    window_size = 30
    if len(last_observations) != window_size:
        return jsonify({'error': 'Provide exactly {} last observations'.format(window_size)}), 400

    # Scale the input data
    scaled_data = scaler.transform(np.array(last_observations).reshape(-1, 1))

    # Prepare the input data for prediction
    future_data = scaled_data[-window_size:, :]

    # Predict the next 7 days
    future_steps = 7
    future_predictions = []
    for _ in range(future_steps):
        future_input = future_data.reshape((1, window_size, 1))
        future_pred = model_bawang_merah.predict(future_input)
        future_predictions.append(future_pred[0, 0])
        future_data = np.vstack((future_data[1:], future_pred))

    # Inverse transform the predictions to the original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

    # Return the predictions as a JSON response
    return jsonify({'predictions': future_predictions})

