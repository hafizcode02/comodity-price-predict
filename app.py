import os
from src.predict.index import predict
from src.web.index import web
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

app.register_blueprint(web)
app.register_blueprint(predict)

if __name__ == "__main__":
    app.run(debug=True, host="localhost" , port=2024)