import os
from src.predict.index import predict
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

app.register_blueprint(predict)

@app.route('/', methods=['GET'])
def home():
    return "Project - Comodity Price Predict"

if __name__ == "__main__":
    app.run(debug=True, host="localhost" , port=2024)