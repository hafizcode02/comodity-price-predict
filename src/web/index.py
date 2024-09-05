from flask import Blueprint, render_template, request, jsonify, send_from_directory, abort
import os
from werkzeug.utils import secure_filename
import pandas as pd

web = Blueprint('web', __name__, template_folder='templates')

@web.route('/')
def index():
    return render_template('index.html', active_page='dashboard')

@web.route('/predict')
def predict():
    return render_template('predict.html', active_page='predict')

@web.route('/config')
def config():
    return render_template('config.html', active_page='config')