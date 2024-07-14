from flask import Blueprint, render_template, request, jsonify
from jinja2 import Template, TemplateNotFound

web = Blueprint('web', __name__, template_folder='templates')

@web.route('/')
def index():
    # return "Hello, World!"
    return render_template('index.html', active_page='dashboard')

@web.route('/predict')
def predict():
    return render_template('predict.html', active_page='predict')