import os
from src.predict.index import predict
from src.web.index import web
from src.comodity.index import comodity
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

app.register_blueprint(web)
app.register_blueprint(predict)
app.register_blueprint(comodity)

# Print all routes
def print_routes():
    for rule in app.url_map.iter_rules():
        print(f"Endpoint: {rule.endpoint}, URL: {rule}")

if __name__ == "__main__":
    print_routes()  # Call function to print routes
    app.run(debug=True, host="localhost" , port=2024)