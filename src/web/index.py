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

# Upload Handler Route
# Uploads the Excel file to the server
UPLOAD_FOLDER = 'storage/'
ALLOWED_EXTENSIONS = {'xlsx'}
SAVED_FILENAME = 'dataset.xlsx'
REQUIRED_COLUMNS = ['date', 'cabai_merah_besar', 'cabai_merah_keriting', 'cabai_rawit_hijau', 'cabai_rawit_merah', 'bawang_merah']
MINIMUM_ROWS = 30

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file_path):
    try:
        df = pd.read_excel(file_path)
        # Check for required columns
        if list(df.columns) != REQUIRED_COLUMNS:
            return False, 'Invalid columns. The file must contain the following columns: ' + ', '.join(REQUIRED_COLUMNS)
        
        # Check for minimum number of rows
        if len(df) < MINIMUM_ROWS:
            return False, 'The file must contain at least {} rows of data excluding the header.'.format(MINIMUM_ROWS)
        
        return True, 'File is valid'
    except Exception as e:
        return False, str(e)
           
@web.route('/upload-xlsx', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        is_valid, message = validate_file(file_path)
        if not is_valid:
            os.remove(file_path)
            return jsonify({'error': message}), 400
        
        os.rename(file_path, os.path.join(UPLOAD_FOLDER, SAVED_FILENAME))
        return jsonify({'message': 'File uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Allowed file type is xlsx'}), 400
    
@web.route('/get-example-dataset', methods=['GET'])
def download_example_dataset():
    try:
        # Ensure the file exists in the storage folder
        if not os.path.isfile(os.path.join(UPLOAD_FOLDER, 'example-dataset.xlsx')):
            abort(404)
        
        # Send the file from the storage folder
        return send_from_directory(UPLOAD_FOLDER, 'example-dataset.xlsx', as_attachment=True)
    except Exception as e:
        return str(e), 500