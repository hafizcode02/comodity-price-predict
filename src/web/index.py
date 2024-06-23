from flask import Blueprint, request, jsonify

web = Blueprint('web', __name__, template_folder='templates')

@web.route('/')
def index():
    return "Hello, World!"